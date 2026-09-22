#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_balanced_testset_sasa_exposure.py -- Builds a balanced test set with:
1. 50% Exposed / 50% Buried (using a SASA threshold)
2. Uniform amino acid distribution within each group
3. Automatically EXCLUDES rare amino acids: P, W, C, M, R
4. Preserves the SS (Secondary Structure) feature

Example:
  python create_balanced_testset.py \
    --input windows_all.json \
    --output windows_testset.json \
    --n_samples 364 \
    --sasa_threshold 0.1199 \
    --remove_from_original \
    --seed 42
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse

# Standard amino acid order
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Amino acids to EXCLUDE (too rare)
EXCLUDED_AA = {'P', 'W', 'C', 'M', 'R'}

# Mapping from 3-letter to 1-letter code
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

# Add case-insensitive variants
for aa3, aa1 in list(AA_3_TO_1.items()):
    AA_3_TO_1[aa3.lower()] = aa1
    AA_3_TO_1[aa3.capitalize()] = aa1

# SS (Secondary Structure) code mapping
SS_MAP = {
    'H': 'Helix', 'E': 'Strand', 'C': 'Coil', 'S': 'Bend',
    'T': 'Turn', 'G': '310helix', 'I': 'Pihelix', 'B': 'Bridge',
    None: 'Missing', ' ': 'Coil', '': 'Coil'
}


def load_windows(path: Path):
    """
    Loads the list of windows from a JSON file.

    Args:
        path: Path to the input JSON file (a list of window dicts).

    Returns:
        The parsed JSON content (typically a list of window dicts).
    """
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def group_by_aa_and_exposure(windows, sasa_threshold=0.13, debug=True):
    """
    Groups windows by (central amino acid, exposure type).
    Automatically EXCLUDES amino acids listed in EXCLUDED_AA.
    Preserves the SS feature for each residue.
    Only windows whose central residue's secondary structure is Strand
    or Coil are kept (see the filter below).

    Args:
        windows: List of window dicts (each with a "residues" list).
        sasa_threshold: Threshold used to define "exposed" (SASA >=
            threshold) vs "buried" (SASA < threshold).
        debug: If True, print diagnostic information about the first
            window and about excluded/unrecognized amino acids.

    Returns:
        tuple:
            grouped (dict): {(aa, 'exposed'|'buried'): [windows]}
            sasa_values (dict): {(aa, exposure): [sasa_values]}
            ss_distributions (dict): {(aa, exposure): {ss_type: count}}
    """
    grouped = defaultdict(list)
    sasa_values = defaultdict(list)
    ss_distributions = defaultdict(lambda: defaultdict(int))
    unrecognized = set()
    excluded_count = defaultdict(int)
    
    # Debug: inspect the first window
    if debug and len(windows) > 0:
        print("\n[DEBUG] Structure of the first window:")
        first = windows[0]
        print(f"  Window keys: {list(first.keys())}")
        if 'residues' in first and len(first['residues']) >= 3:
            res2 = first['residues'][2]
            print(f"  Residue[2] (central):")
            print(f"    AA: {res2.get('AA')}")
            print(f"    ACC_norm: {res2.get('ACC_norm')}")
            print(f"    SS: {res2.get('SS', 'N/A')}")
    
    for window in windows:
        residues = window.get('residues', [])
        
        # The central residue is at position 2 (0-indexed: 0,1,2,3,4)
        if len(residues) >= 5:
            central_residue = residues[2]
            
            # Extract AA
            central_aa_3letter = (
                central_residue.get('AA') or
                central_residue.get('aa') or
                central_residue.get('ResName') or
                central_residue.get('resname') or
                ''
            )
            central_aa = AA_3_TO_1.get(central_aa_3letter, '')
            
            # Extract normalized SASA
            sasa = central_residue.get('ACC_norm')
            
            # Extract SS
            ss = central_residue.get('SS')
            ss_label = SS_MAP.get(ss, 'Coil')  # Default to Coil if empty

            # FILTER: keep only E and C (Strand and Coil)
            if ss_label not in ['Strand', 'Coil']:
                continue
            
            # Filter out excluded amino acids
            if central_aa in EXCLUDED_AA:
                excluded_count[central_aa] += 1
                continue
            
            # Determine exposure type
            if sasa is not None:
                exposure_type = 'exposed' if sasa >= sasa_threshold else 'buried'
            else:
                continue  # Skip if SASA is not available
            
            # Group only valid amino acids
            if central_aa and central_aa in AA_ORDER:
                key = (central_aa, exposure_type)
                grouped[key].append(window)
                sasa_values[key].append(float(sasa))
                
                # Count SS distribution
                ss_label = SS_MAP.get(ss, 'Missing')
                ss_distributions[key][ss_label] += 1
                
            elif central_aa_3letter and central_aa_3letter not in unrecognized:
                unrecognized.add(central_aa_3letter)
    
    if debug:
        if excluded_count:
            print(f"\n[DEBUG] Amino acids automatically EXCLUDED:")
            for aa, count in sorted(excluded_count.items()):
                print(f"  {aa}: {count} windows -> REMAIN IN TRAINING SET")
        if unrecognized:
            print(f"\n[DEBUG] Unrecognized amino acids: {unrecognized}")
    
    return grouped, sasa_values, ss_distributions


def select_balanced_samples(grouped, sasa_values, ss_distributions, n_total=300, seed=42):
    """
    Selects a balanced set of samples such that:
    1. 50% exposed / 50% buried
    2. Uniform amino acid distribution within each group
    3. The natural SS distribution is preserved

    Args:
        grouped: dict {(aa, exposure): [windows]}, as returned by
            `group_by_aa_and_exposure`.
        sasa_values: dict {(aa, exposure): [sasa_values]}, as returned by
            `group_by_aa_and_exposure`.
        ss_distributions: dict {(aa, exposure): {ss_type: count}}, as
            returned by `group_by_aa_and_exposure`.
        n_total: Total number of samples to select (split ~50/50
            exposed/buried).
        seed: Random seed for reproducibility.

    Returns:
        tuple:
            selected (list): The list of selected window dicts.
            stats (dict): {(aa, exposure): {...}} per-group statistics
                (availability, selection count, SASA mean/std/min/max,
                SS distribution, exclusion reason if any).
    """
    np.random.seed(seed)
    
    # Determine which amino acids are available
    available_aa = sorted([aa for aa in AA_ORDER if aa not in EXCLUDED_AA])
    n_aa_available = len(available_aa)
    
    if n_aa_available == 0:
        print("[WARN] No amino acid available!")
        return [], {}
    
    print(f"\nAmino acids INCLUDED in the test set: {n_aa_available}/{len(AA_ORDER)}")
    print(f"  Included AAs: {', '.join(available_aa)}")
    print(f"  Excluded AAs: {', '.join(sorted(EXCLUDED_AA))}")
    
    # 50/50 split between exposed and buried
    n_exposed = n_total // 2
    n_buried = n_total - n_exposed
    
    print(f"\nBalancing strategy:")
    print(f"  Target EXPOSED: {n_exposed} samples")
    print(f"  Target BURIED: {n_buried} samples")
    print(f"  Samples per AA per group: ~{n_exposed // n_aa_available} exposed, ~{n_buried // n_aa_available} buried")
    
    selected = []
    stats = {}
    
    # Select for each group (exposed/buried)
    for exposure_type, n_target in [('exposed', n_exposed), ('buried', n_buried)]:
        samples_per_aa = n_target // n_aa_available
        remainder = n_target % n_aa_available
        
        print(f"\n--- Selecting {exposure_type.upper()} ---")
        print(f"  Samples per AA: {samples_per_aa}")
        print(f"  Extra samples to distribute: {remainder}")
        
        # Phase 1: distribute evenly across amino acids
        aa_selections = {}
        
        for aa in AA_ORDER:
            key = (aa, exposure_type)
            
            if aa in EXCLUDED_AA:
                if key not in stats:
                    stats[key] = {
                        'aa': aa,
                        'exposure': exposure_type,
                        'available': 0,
                        'selected': 0,
                        'excluded': True,
                        'reason': 'Rare AA excluded',
                        'sasa_mean': None,
                        'sasa_std': None,
                        'sasa_min': None,
                        'sasa_max': None,
                        'ss_distribution': {}
                    }
                continue
            
            windows = grouped.get(key, [])
            n_available = len(windows)
            sasa_vals = sasa_values.get(key, [])
            ss_dist = ss_distributions.get(key, {})
            
            stats[key] = {
                'aa': aa,
                'exposure': exposure_type,
                'available': n_available,
                'selected': 0,
                'excluded': False,
                'sasa_mean': np.mean(sasa_vals) if sasa_vals else None,
                'sasa_std': np.std(sasa_vals) if sasa_vals else None,
                'sasa_min': np.min(sasa_vals) if sasa_vals else None,
                'sasa_max': np.max(sasa_vals) if sasa_vals else None,
                'ss_distribution': dict(ss_dist)
            }
            
            if n_available == 0:
                continue
            
            # Select samples for this amino acid
            n_select = min(samples_per_aa, n_available)
            
            # Random selection that preserves the natural SASA/SS distribution
            selected_indices = np.random.choice(n_available, n_select, replace=False)
            selected_windows = [windows[i] for i in selected_indices]
            
            selected.extend(selected_windows)
            stats[key]['selected'] = n_select
            aa_selections[aa] = {
                'windows': windows,
                'selected_count': n_select,
                'available': n_available
            }
        
        # Phase 2: distribute the remainder to amino acids with more samples available
        if remainder > 0:
            # Build a list of amino acid candidates that can receive extra samples
            candidates = []
            for aa in available_aa:
                key = (aa, exposure_type)
                sel_info = aa_selections.get(aa)
                if sel_info:
                    avail = sel_info['available']
                    selected_count = sel_info['selected_count']
                    
                    if selected_count < avail:
                        can_add = avail - selected_count
                        candidates.append((aa, can_add))
            
            # Sort by remaining capacity
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Distribute cyclically
            for i in range(remainder):
                if not candidates:
                    break
                
                aa, _ = candidates[i % len(candidates)]
                key = (aa, exposure_type)
                windows = grouped[key]
                
                # Pick one additional sample not yet selected
                idx = np.random.choice(len(windows))
                attempts = 0
                while windows[idx] in selected and attempts < 100:
                    idx = np.random.choice(len(windows))
                    attempts += 1
                
                if windows[idx] not in selected:
                    selected.append(windows[idx])
                    stats[key]['selected'] += 1
    
    return selected, stats


def print_statistics(stats, selected, sasa_threshold):
    """
    Prints statistics about the selection, including SASA, exposure and
    SS information.

    Args:
        stats: Per-(AA, exposure) statistics dict, as returned by
            `select_balanced_samples`.
        selected: List of selected window dicts.
        sasa_threshold: SASA threshold used to define exposed vs buried
            (only used for reporting/labeling).
    """
    print("\n" + "="*140)
    print("SELECTION STATISTICS (AA + EXPOSURE + SS BALANCING)")
    print("="*140)
    print(f"{'AA':<4} {'Exposure':<8} {'Available':<12} {'Selected':<12} {'SASA (mean+/-std)':<25} {'SASA Range':<20} {'SS Dominant':<15} {'Note':<15}")
    print("-"*140)
    
    # Counters
    total_exposed = 0
    total_buried = 0
    selected_per_aa_exposed = []
    selected_per_aa_buried = []
    
    for aa in AA_ORDER:
        for exposure in ['exposed', 'buried']:
            key = (aa, exposure)
            info = stats.get(key, {})
            
            avail = info.get('available', 0)
            sel = info.get('selected', 0)
            sasa_mean = info.get('sasa_mean')
            sasa_std = info.get('sasa_std')
            sasa_min = info.get('sasa_min')
            sasa_max = info.get('sasa_max')
            ss_dist = info.get('ss_distribution', {})
            
            # Format SASA stats
            if sasa_mean is not None:
                sasa_stats = f"{sasa_mean:.3f}+/-{sasa_std:.3f}"
                sasa_range = f"{sasa_min:.3f}-{sasa_max:.3f}"
            else:
                sasa_stats = "-"
                sasa_range = "-"
            
            # Find dominant SS
            if ss_dist:
                dominant_ss = max(ss_dist.items(), key=lambda x: x[1])
                ss_info = f"{dominant_ss[0]}({dominant_ss[1]})"
            else:
                ss_info = "-"
            
            if info.get('reason'):
                note = "EXCLUDED"
            elif sel > 0:
                note = "Included"
            else:
                note = "-"
            
            if exposure == 'exposed':
                total_exposed += sel
                if sel > 0 and aa not in EXCLUDED_AA:
                    selected_per_aa_exposed.append(sel)
            else:
                total_buried += sel
                if sel > 0 and aa not in EXCLUDED_AA:
                    selected_per_aa_buried.append(sel)
            
            print(f"{aa:<4} {exposure:<8} {avail:<12} {sel:<12} {sasa_stats:<25} {sasa_range:<20} {ss_info:<15} {note:<15}")
    
    print("-"*140)
    print(f"{'TOT':<4} {'EXPOSED':<8} {'':<12} {total_exposed:<12}")
    print(f"{'TOT':<4} {'BURIED':<8} {'':<12} {total_buried:<12}")
    print(f"{'TOT':<4} {'ALL':<8} {'':<12} {total_exposed + total_buried:<12}")
    print("="*140)
    
    # EXPOSURE balancing statistics
    print(f"\nEXPOSURE BALANCE (SASA threshold={sasa_threshold}):")
    print(f"  Exposed: {total_exposed} ({total_exposed/(total_exposed+total_buried)*100:.1f}%)")
    print(f"  Buried:  {total_buried} ({total_buried/(total_exposed+total_buried)*100:.1f}%)")
    
    exposure_imbalance = abs(total_exposed - total_buried)
    if exposure_imbalance <= 2:
        print("  EXPOSURE balance: excellent")
    elif exposure_imbalance <= 10:
        print("  EXPOSURE balance: acceptable")
    else:
        print("  EXPOSURE balance: imbalanced")
    
    # AA balancing statistics (per group)
    for exposure_type, samples in [('EXPOSED', selected_per_aa_exposed), 
                                    ('BURIED', selected_per_aa_buried)]:
        if samples:
            print(f"\nAA BALANCE ({exposure_type}, rare AAs excluded):")
            print(f"  Min samples: {min(samples)}")
            print(f"  Max samples: {max(samples)}")
            print(f"  Mean: {np.mean(samples):.2f}")
            print(f"  Std dev: {np.std(samples):.2f}")
            print(f"  Range: {max(samples) - min(samples)}")
            
            aa_imbalance = max(samples) - min(samples)
            if aa_imbalance <= 2:
                print(f"  AA balance in {exposure_type}: excellent")
            elif aa_imbalance <= 5:
                print(f"  AA balance in {exposure_type}: acceptable")
            else:
                print(f"  AA balance in {exposure_type}: imbalanced")
    
    # Overall SASA distribution in the test set
    print(f"\nOVERALL SASA DISTRIBUTION IN THE TEST SET:")
    all_sasa = []
    for window in selected:
        residues = window.get('residues', [])
        if len(residues) >= 5:
            sasa = residues[2].get('ACC_norm')
            if sasa is not None:
                all_sasa.append(float(sasa))
    
    if all_sasa:
        print(f"  SASA mean: {np.mean(all_sasa):.3f}")
        print(f"  SASA std: {np.std(all_sasa):.3f}")
        print(f"  SASA min: {np.min(all_sasa):.3f}")
        print(f"  SASA max: {np.max(all_sasa):.3f}")
        print(f"  SASA median: {np.median(all_sasa):.3f}")
        
        # Check the distribution relative to the threshold
        n_above_threshold = sum(1 for s in all_sasa if s >= sasa_threshold)
        n_below_threshold = len(all_sasa) - n_above_threshold
        print(f"  Samples with SASA >= {sasa_threshold}: {n_above_threshold} ({n_above_threshold/len(all_sasa)*100:.1f}%)")
        print(f"  Samples with SASA < {sasa_threshold}: {n_below_threshold} ({n_below_threshold/len(all_sasa)*100:.1f}%)")
    
    # SS distribution in the test set
    print(f"\nSECONDARY STRUCTURE DISTRIBUTION IN THE TEST SET:")
    ss_counts = defaultdict(int)
    for window in selected:
        residues = window.get('residues', [])
        if len(residues) >= 5:
            ss = residues[2].get('SS')
            ss_label = SS_MAP.get(ss, 'Missing')
            ss_counts[ss_label] += 1
    
    if ss_counts:
        total_ss = sum(ss_counts.values())
        for ss_type in sorted(ss_counts.keys()):
            count = ss_counts[ss_type]
            print(f"  {ss_type:10s}: {count:4d} ({count/total_ss*100:5.1f}%)")


def save_to_csv(windows, output_path):
    """
    Saves a list of window dicts to a CSV file.

    Args:
        windows: List of window dicts to save.
        output_path: Path to the output CSV file.
    """
    df = pd.DataFrame(windows)
    df.to_csv(output_path, index=False)
    print(f"CSV saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Build a test set balanced by amino acid and EXPOSURE (via a SASA threshold), preserving SS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python create_balanced_testset.py \\
    --input windows_all.json \\
    --output windows_testset_balanced.json \\
    --n_samples 300 \\
    --sasa_threshold 0.1199 \\
    --remove_from_original \\
    --seed 42

Notes:
  - Balances on 50% exposed / 50% buried (using the SASA threshold)
  - Within each group, the amino acid distribution is uniform
  - Preserves the natural SASA and SS distribution for each (AA, exposure) pair
  - Automatically EXCLUDES rare amino acids: P, W, C, M, R
  - Excluded amino acids remain in the training set
  - Preserves the SS (Secondary Structure) feature for each residue
        """
    )
    ap.add_argument("--input", required=True, 
                    help="Input JSON file with the windows")
    ap.add_argument("--output", required=True,
                    help="Output JSON file for the balanced test set")
    ap.add_argument("--output_csv", 
                    help="Output CSV file (default: same name as the JSON output, with .csv extension)")
    ap.add_argument("--n_samples", type=int, default=300,
                    help="Total number of samples to select (default: 300)")
    ap.add_argument("--sasa_threshold", type=float, default=0.1199,
                    help="SASA threshold used to define exposed (>= threshold) vs buried (< threshold)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    ap.add_argument("--remove_from_original", action="store_true",
                    help="Remove the selected samples from the original input file")
    
    args = ap.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Determine the CSV output path
    if args.output_csv:
        output_csv_path = Path(args.output_csv)
    else:
        output_csv_path = output_path.with_suffix('.csv')
    
    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path}")
    
    print("="*140)
    print("LOADING DATA - BALANCING ON AA + EXPOSURE (SASA THRESHOLD) + SS")
    print("="*140)
    print(f"Input: {input_path}")
    print(f"Output JSON: {output_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Requested N samples: {args.n_samples}")
    print(f"SASA threshold: {args.sasa_threshold}")
    print(f"Seed: {args.seed}")
    
    # Load windows
    windows = load_windows(input_path)
    print(f"Total windows: {len(windows)}")
    
    # Group by (AA, exposure)
    print("\nGrouping by (amino acid, exposure) with SS analysis...")
    grouped, sasa_values, ss_distributions = group_by_aa_and_exposure(
        windows, 
        sasa_threshold=args.sasa_threshold,
        debug=True
    )
    
    print(f"\n(AA, exposure) combinations found: {len(grouped)}")
    
    # Availability analysis by exposure
    exposed_total = sum(len(v) for k, v in grouped.items() if k[1] == 'exposed')
    buried_total = sum(len(v) for k, v in grouped.items() if k[1] == 'buried')
    
    print(f"\nDistribution in the original dataset:")
    print(f"  EXPOSED (SASA >= {args.sasa_threshold}): {exposed_total} windows ({exposed_total/(exposed_total+buried_total)*100:.1f}%)")
    print(f"  BURIED (SASA < {args.sasa_threshold}): {buried_total} windows ({buried_total/(exposed_total+buried_total)*100:.1f}%)")
    
    # Select balanced samples
    print("\n" + "="*140)
    print("BALANCED SELECTION")
    print("="*140)
    
    selected, stats = select_balanced_samples(
        grouped, sasa_values, ss_distributions,
        args.n_samples, args.seed
    )
    
    # Statistics
    print_statistics(stats, selected, args.sasa_threshold)
    
    # Save JSON output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    
    print(f"\nBalanced test set saved: {output_path}")
    print(f"   Selected samples: {len(selected)}")
    
    # Save CSV output
    save_to_csv(selected, output_csv_path)
    
    # Remove from the original file if requested
    if args.remove_from_original:
        print("\nRemoving selected samples from the original file...")
        
        # Build the set of selected IDs
        selected_ids = {w.get('window_id') for w in selected if 'window_id' in w}
        
        # Keep windows that were not selected
        remaining = [w for w in windows if w.get('window_id') not in selected_ids]
        
        with input_path.open('w', encoding='utf-8') as f:
            json.dump(remaining, f, ensure_ascii=False, indent=2)
        
        print(f"Original file updated: {input_path}")
        print(f"   Remaining windows: {len(remaining)}")
        print(f"   Removed windows: {len(windows) - len(remaining)}")
        print(f"\nNote: the excluded amino acids ({', '.join(sorted(EXCLUDED_AA))}) all remained in the training set")


if __name__ == "__main__":
    main()
