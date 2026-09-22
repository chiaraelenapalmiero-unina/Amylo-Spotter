#!/usr/bin/env python3
"""
Script to convert cluster indices from GLOBAL (0-97) to LOCAL (0-N for
each amino acid).
Updated version for datasets with continuous SASA and SS (Secondary
Structure) fields.

Usage:
    python convert_global_to_local_clusters.py \
        --input ./windows_all.npz \
        --output ./windows_all_local.npz \
        --rotamer_csv rotamer_localmax_maxdens.csv \
        --verify
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# rotamer_library.py lives in ../common (a dependency shared with model/)
sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from rotamer_library import load_rotamer_library, AA20

def convert_global_to_local(aa_ids, cluster_global, rotamer_lib):
    """
    Converts cluster indices from global to local.

    Args:
        aa_ids: Array of shape [N, L] with amino acid indices.
        cluster_global: Array of shape [N, L] with GLOBAL cluster indices
            (0-97).
        rotamer_lib: A RotamerLibrary instance, used to look up which
            global index corresponds to which (AA, local cluster) pair.

    Returns:
        cluster_local: Array of shape [N, L] with LOCAL cluster indices
            (0-n for each amino acid).
        conversion_stats: Dict with conversion statistics:
            'total'     - total number of (residue) positions processed
            'converted' - positions successfully mapped from a known
                          global index to its local index
            'invalid'   - positions whose global index did not match any
                          known (AA, cluster) pair for that AA; these are
                          mapped with a modulo fallback (see below)
            'unchanged' - reserved counter, currently always 0
            'mask'      - positions skipped because they are MASK tokens
                          (AA index >= 20) or a MASK cluster (global index
                          >= 15)
    """
    cluster_local = cluster_global.copy()
    
    # Conversion statistics
    stats = {
        'total': 0,
        'converted': 0,
        'invalid': 0,
        'unchanged': 0,
        'mask': 0
    }
    
    N, L = aa_ids.shape
    
    for i in range(N):
        for j in range(L):
            stats['total'] += 1
            
            aa_idx = int(aa_ids[i, j])
            cl_global = int(cluster_global[i, j])
            
            # Skip MASK tokens (AA >= 20)
            if aa_idx >= 20:
                stats['mask'] += 1
                continue
            
            # Skip MASK clusters (we assume very high cluster indices are mask tokens)
            if cl_global >= 15:  # Reasonable threshold
                stats['mask'] += 1
                continue
            
            # Check whether this global cluster corresponds to this AA
            # in the rotamer library
            found = False
            
            for local_idx in range(rotamer_lib.get_n_clusters(aa_idx)):
                global_idx = rotamer_lib.pair_to_global.get((aa_idx, local_idx), -1)
                
                if global_idx == cl_global:
                    # Found it! Convert.
                    cluster_local[i, j] = local_idx
                    stats['converted'] += 1
                    found = True
                    break
            
            if not found:
                # Not found - this global cluster does not belong to this AA.
                # Try to map it anyway using the remainder.
                n_clusters = rotamer_lib.get_n_clusters(aa_idx)
                if n_clusters > 0:
                    # Fallback: map using modulo
                    local_fallback = cl_global % n_clusters
                    cluster_local[i, j] = local_fallback
                    stats['invalid'] += 1
                else:
                    stats['invalid'] += 1
    
    return cluster_local, stats


def verify_conversion(aa_ids, cluster_local, rotamer_lib):
    """
    Verifies that, after conversion, every (AA, local cluster) pair in
    the dataset is a valid pair according to the rotamer library.

    Args:
        aa_ids: Array of shape [N, L] with amino acid indices.
        cluster_local: Array of shape [N, L] with LOCAL cluster indices,
            as produced by `convert_global_to_local`.
        rotamer_lib: A RotamerLibrary instance used to check validity.

    Returns:
        tuple (valid_count, invalid_count): Number of non-masked
        positions whose (AA, cluster) pair is valid / invalid. Prints the
        first 5 invalid pairs found, as a warning, for quick debugging.
    """
    N, L = aa_ids.shape
    invalid_count = 0
    valid_count = 0
    
    for i in range(N):
        for j in range(L):
            aa_idx = int(aa_ids[i, j])
            cl_idx = int(cluster_local[i, j])
            
            # Skip MASK
            if aa_idx >= 20:
                continue
            
            valid_clusters = rotamer_lib.get_valid_clusters(aa_idx)
            if cl_idx in valid_clusters:
                valid_count += 1
            else:
                invalid_count += 1
                if invalid_count <= 5:  # Print only the first 5
                    aa_name = AA20[aa_idx] if aa_idx < 20 else "?"
                    print(f"  [WARN] Still invalid: AA={aa_idx} ({aa_name}), cluster={cl_idx}, "
                          f"valid={valid_clusters}")
    
    return valid_count, invalid_count


def analyze_ss_distribution(ss_ids):
    """
    Analyzes and prints the distribution of secondary-structure classes
    in the dataset.

    Args:
        ss_ids: Array of integer SS class ids (any shape; flattened for
            the count).
    """
    ss_names = {
        0: 'Helix', 1: 'Strand', 2: 'Coil', 3: 'Bend',
        4: 'Turn', 5: '310helix', 6: 'Pihelix', 7: 'Bridge', 8: 'Missing'
    }
    
    unique, counts = np.unique(ss_ids, return_counts=True)
    total = ss_ids.size
    
    print("\nSecondary Structure Distribution:")
    print("-" * 60)
    for ss_id, count in zip(unique, counts):
        ss_name = ss_names.get(ss_id, f'Unknown({ss_id})')
        print(f"  {ss_name:10s} (ID {ss_id}): {count:8d} ({count/total*100:5.1f}%)")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convert cluster indices from global to local (with SS support)")
    parser.add_argument("--input", required=True, help="Path to the input .npz file (must contain at least 'aa_ids' and 'cluster_idx' arrays with GLOBAL cluster indices).")
    parser.add_argument("--output", required=True, help="Path to the output .npz file to write, identical to the input but with 'cluster_idx' converted to LOCAL indices.")
    parser.add_argument("--rotamer_csv", default="rotamer_localmax_maxdens.csv", 
                       help="Path to the rotamer library CSV file used to look up valid (AA, cluster) pairs (default: rotamer_localmax_maxdens.csv).")
    parser.add_argument("--verify", action="store_true", 
                       help="After conversion, re-check every (AA, cluster) pair against the rotamer library and report how many are valid.")
    parser.add_argument("--analyze_ss", action="store_true",
                       help="If the dataset contains an 'ss' (Secondary Structure) array, print its class distribution.")
    
    args = parser.parse_args()
    
    # Load rotamer library
    print("="*60)
    print("LOADING ROTAMER LIBRARY")
    print("="*60)
    lib = load_rotamer_library(args.rotamer_csv)
    
    # Load dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    print(f"Input file: {args.input}")
    
    data = np.load(args.input, allow_pickle=True)  # allow_pickle is needed for the pdb_id string array
    aa_ids = data["aa_ids"]
    cluster_global = data["cluster_idx"]
    
    print(f"Shape: {aa_ids.shape}")
    print(f"AA range: {aa_ids.min()} - {aa_ids.max()}")
    print(f"Cluster range (global): {cluster_global.min()} - {cluster_global.max()}")
    
    # Check whether SASA is present
    if "sasa" in data:
        print(f"SASA found - range: {data['sasa'].min():.4f} - {data['sasa'].max():.4f}")
        print(f"SASA mean: {data['sasa'].mean():.4f}")
        print(f"SASA std: {data['sasa'].std():.4f}")
    elif "exposure" in data:
        print(f"Exposure found (binary) - using legacy format")
    
    # Check whether SS is present
    has_ss = "ss" in data
    if has_ss:
        ss_ids = data["ss"]
        print(f"SS (Secondary Structure) found - range: {ss_ids.min()} - {ss_ids.max()}")
        print(f"SS shape: {ss_ids.shape}")
        
        if args.analyze_ss:
            analyze_ss_distribution(ss_ids)
    else:
        print("[WARN] No Secondary Structure (SS) data found in dataset")
    
    # Check whether pos_idx is present
    if "pos_idx" in data:
        pos_idx = data["pos_idx"]
        print(f"Position indices found - range: {pos_idx.min()} - {pos_idx.max()}")
    
    # Check whether protein_class_id is present
    if "protein_class_id" in data:
        protein_class_ids = data["protein_class_id"]
        print(f"Protein Class IDs found - range: {protein_class_ids.min()} - {protein_class_ids.max()}")
        unique_classes = np.unique(protein_class_ids)
        print(f"  Unique protein classes: {len(unique_classes)}")
    
    # Check whether pdb_id is present
    if "pdb_id" in data:
        pdb_ids = data["pdb_id"]
        print(f"PDB IDs found - total: {len(pdb_ids)}")
        unique_pdbs = len(set(pdb_ids))
        print(f"  Unique PDB IDs: {unique_pdbs}")
    
    # Convert
    print("\n" + "="*60)
    print("CONVERTING CLUSTERS")
    print("="*60)
    
    cluster_local, stats = convert_global_to_local(aa_ids, cluster_global, lib)
    
    print(f"Total positions: {stats['total']}")
    print(f"  Converted:     {stats['converted']} ({stats['converted']/stats['total']*100:.1f}%)")
    print(f"  Invalid (fallback): {stats['invalid']} ({stats['invalid']/stats['total']*100:.1f}%)")
    print(f"  Unchanged:     {stats['unchanged']} ({stats['unchanged']/stats['total']*100:.1f}%)")
    print(f"  Masked:        {stats['mask']} ({stats['mask']/stats['total']*100:.1f}%)")
    
    print(f"\nCluster range (local): {cluster_local.min()} - {cluster_local.max()}")
    
    # Verify
    if args.verify:
        print("\n" + "="*60)
        print("VERIFYING CONVERSION")
        print("="*60)
        
        valid, invalid = verify_conversion(aa_ids, cluster_local, lib)
        total = valid + invalid
        
        if total > 0:
            print(f"Valid pairs:   {valid}/{total} ({valid/total*100:.1f}%)")
            print(f"Invalid pairs: {invalid}/{total} ({invalid/total*100:.1f}%)")
            
            if invalid == 0:
                print("\nALL PAIRS ARE NOW VALID!")
            else:
                print(f"\n[WARN] Still {invalid} invalid pairs. Check warnings above.")
        else:
            print("No non-masked pairs to validate.")
    
    # Save
    print("\n" + "="*60)
    print("SAVING CONVERTED DATASET")
    print("="*60)
    
    # Copy over all other fields (including SASA and SS, if present)
    output_data = {}
    for key in data.keys():
        if key == "cluster_idx":
            output_data[key] = cluster_local
        else:
            output_data[key] = data[key]
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path, **output_data)
    
    print(f"Saved to: {args.output}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Show saved fields
    print(f"\nFields saved:")
    for key in output_data.keys():
        arr = output_data[key]
        if hasattr(arr, 'shape'):
            print(f"  - {key}: {arr.shape} ({arr.dtype})")
        else:
            print(f"  - {key}: {type(arr)}")
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nYou can now use: {args.output}")
    print("Remember to convert both train and test datasets!")
    
    if has_ss:
        print("\nSecondary Structure (SS) data preserved in output")
    else:
        print("\n[WARN] No SS data in input - consider regenerating dataset with SS features")


if __name__ == "__main__":
    main()
