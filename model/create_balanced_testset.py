#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_balanced_testset_sasa_exposure.py — Crea un test set bilanciato per:
1. 50% Exposed / 50% Buried (usando SASA threshold)
2. Distribuzione uniforme degli AA in ciascun gruppo
3. ESCLUDE automaticamente gli AA rari: P, W, C, M, R
4. Preserva la feature SS (Secondary Structure)

Esempio:
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

# Ordine standard degli amminoacidi
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# AA da ESCLUDERE (troppo rari)
EXCLUDED_AA = {'P', 'W', 'C', 'M', 'R'}

# Mappatura da codice 3 lettere a 1 lettera
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

# Aggiungi varianti case-insensitive
for aa3, aa1 in list(AA_3_TO_1.items()):
    AA_3_TO_1[aa3.lower()] = aa1
    AA_3_TO_1[aa3.capitalize()] = aa1

# Mappatura SS
SS_MAP = {
    'H': 'Helix', 'E': 'Strand', 'C': 'Coil', 'S': 'Bend',
    'T': 'Turn', 'G': '310helix', 'I': 'Pihelix', 'B': 'Bridge',
    None: 'Missing', ' ': 'Coil', '': 'Coil'
}


def load_windows(path: Path):
    """Carica le finestre dal file JSON"""
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def group_by_aa_and_exposure(windows, sasa_threshold=0.13, debug=True):
    """
    Raggruppa le finestre per (AA, exposure_type).
    ESCLUDE automaticamente gli AA in EXCLUDED_AA.
    Preserva la feature SS per ogni residuo.
    
    Args:
        sasa_threshold: Soglia per definire exposed (>= threshold)
    
    Returns:
        dict: {(aa, 'exposed'|'buried'): [windows]},
        dict: {(aa, exposure): [sasa_values]},
        dict: {(aa, exposure): [ss_distributions]}
    """
    grouped = defaultdict(list)
    sasa_values = defaultdict(list)
    ss_distributions = defaultdict(lambda: defaultdict(int))
    unrecognized = set()
    excluded_count = defaultdict(int)
    
    # Debug: analizza la prima finestra
    if debug and len(windows) > 0:
        print("\n[DEBUG] Struttura prima finestra:")
        first = windows[0]
        print(f"  Window keys: {list(first.keys())}")
        if 'residues' in first and len(first['residues']) >= 3:
            res2 = first['residues'][2]
            print(f"  Residue[2] (centrale):")
            print(f"    AA: {res2.get('AA')}")
            print(f"    ACC_norm: {res2.get('ACC_norm')}")
            print(f"    SS: {res2.get('SS', 'N/A')}")
    
    for window in windows:
        residues = window.get('residues', [])
        
        # Il residuo centrale è quello in posizione 2 (0-indexed: 0,1,2,3,4)
        if len(residues) >= 5:
            central_residue = residues[2]
            
            # Estrai AA
            central_aa_3letter = (
                central_residue.get('AA') or
                central_residue.get('aa') or
                central_residue.get('ResName') or
                central_residue.get('resname') or
                ''
            )
            central_aa = AA_3_TO_1.get(central_aa_3letter, '')
            
            # Estrai SASA normalizzato
            sasa = central_residue.get('ACC_norm')
            
            # Estrai SS
            ss = central_residue.get('SS')
            ss_label = SS_MAP.get(ss, 'Coil') # Default a Coil se vuoto

            # FILTRO: Prendi solo E e C (Strand e Coil)
            if ss_label not in ['Strand', 'Coil']:
                continue
            
            # Filtra AA esclusi
            if central_aa in EXCLUDED_AA:
                excluded_count[central_aa] += 1
                continue
            
            # Determina exposure type
            if sasa is not None:
                exposure_type = 'exposed' if sasa >= sasa_threshold else 'buried'
            else:
                continue  # Skip se SASA non disponibile
            
            # Raggruppa solo AA validi
            if central_aa and central_aa in AA_ORDER:
                key = (central_aa, exposure_type)
                grouped[key].append(window)
                sasa_values[key].append(float(sasa))
                
                # Conta distribuzione SS
                ss_label = SS_MAP.get(ss, 'Missing')
                ss_distributions[key][ss_label] += 1
                
            elif central_aa_3letter and central_aa_3letter not in unrecognized:
                unrecognized.add(central_aa_3letter)
    
    if debug:
        if excluded_count:
            print(f"\n[DEBUG] AA ESCLUSI automaticamente:")
            for aa, count in sorted(excluded_count.items()):
                print(f"  {aa}: {count} finestre → RIMANGONO NEL TRAINING")
        if unrecognized:
            print(f"\n[DEBUG] Amminoacidi non riconosciuti: {unrecognized}")
    
    return grouped, sasa_values, ss_distributions


def select_balanced_samples(grouped, sasa_values, ss_distributions, n_total=300, seed=42):
    """
    Seleziona campioni bilanciati per:
    1. 50% exposed / 50% buried
    2. Distribuzione uniforme degli AA in ciascun gruppo
    3. Preserva la distribuzione naturale di SS
    
    Args:
        grouped: dict {(aa, exposure): [windows]}
        sasa_values: dict {(aa, exposure): [sasa_values]}
        ss_distributions: dict {(aa, exposure): {ss_type: count}}
        n_total: numero totale di campioni desiderati
        seed: random seed
    
    Returns:
        list di finestre selezionate, dict con statistiche
    """
    np.random.seed(seed)
    
    # Determina quali AA sono disponibili
    available_aa = sorted([aa for aa in AA_ORDER if aa not in EXCLUDED_AA])
    n_aa_available = len(available_aa)
    
    if n_aa_available == 0:
        print("⚠️  Nessun amminoacido disponibile!")
        return [], {}
    
    print(f"\nAmminoacidi INCLUSI nel test: {n_aa_available}/{len(AA_ORDER)}")
    print(f"  AA inclusi: {', '.join(available_aa)}")
    print(f"  AA esclusi: {', '.join(sorted(EXCLUDED_AA))}")
    
    # Split 50/50 tra exposed e buried
    n_exposed = n_total // 2
    n_buried = n_total - n_exposed
    
    print(f"\nStrategia di bilanciamento:")
    print(f"  Target EXPOSED: {n_exposed} campioni")
    print(f"  Target BURIED: {n_buried} campioni")
    print(f"  Campioni per AA per gruppo: ~{n_exposed // n_aa_available} exposed, ~{n_buried // n_aa_available} buried")
    
    selected = []
    stats = {}
    
    # Seleziona per ogni gruppo (exposed/buried)
    for exposure_type, n_target in [('exposed', n_exposed), ('buried', n_buried)]:
        samples_per_aa = n_target // n_aa_available
        remainder = n_target % n_aa_available
        
        print(f"\n--- Selezione {exposure_type.upper()} ---")
        print(f"  Campioni per AA: {samples_per_aa}")
        print(f"  Campioni extra da distribuire: {remainder}")
        
        # Fase 1: distribuisci equamente tra AA
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
                        'reason': 'AA raro escluso',
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
            
            # Seleziona campioni per questo AA
            n_select = min(samples_per_aa, n_available)
            
            # Selezione random che mantiene la distribuzione naturale di SASA e SS
            selected_indices = np.random.choice(n_available, n_select, replace=False)
            selected_windows = [windows[i] for i in selected_indices]
            
            selected.extend(selected_windows)
            stats[key]['selected'] = n_select
            aa_selections[aa] = {
                'windows': windows,
                'selected_count': n_select,
                'available': n_available
            }
        
        # Fase 2: distribuisci i remainder agli AA con più campioni disponibili
        if remainder > 0:
            # Crea lista di candidati AA che possono ricevere più campioni
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
            
            # Ordina per capacità residua
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Distribuisci ciclicamente
            for i in range(remainder):
                if not candidates:
                    break
                
                aa, _ = candidates[i % len(candidates)]
                key = (aa, exposure_type)
                windows = grouped[key]
                
                # Prendi un campione aggiuntivo non ancora selezionato
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
    """Stampa statistiche sulla selezione con info SASA, exposure e SS"""
    print("\n" + "="*140)
    print("STATISTICHE SELEZIONE (BILANCIAMENTO AA + EXPOSURE + SS)")
    print("="*140)
    print(f"{'AA':<4} {'Exposure':<8} {'Disponibili':<12} {'Selezionati':<12} {'SASA (mean±std)':<25} {'Range SASA':<20} {'SS Dominant':<15} {'Note':<15}")
    print("-"*140)
    
    # Contatori
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
            
            # Formatta SASA stats
            if sasa_mean is not None:
                sasa_stats = f"{sasa_mean:.3f}±{sasa_std:.3f}"
                sasa_range = f"{sasa_min:.3f}-{sasa_max:.3f}"
            else:
                sasa_stats = "-"
                sasa_range = "-"
            
            # Trova SS dominante
            if ss_dist:
                dominant_ss = max(ss_dist.items(), key=lambda x: x[1])
                ss_info = f"{dominant_ss[0]}({dominant_ss[1]})"
            else:
                ss_info = "-"
            
            if info.get('reason'):
                note = "❌ ESCLUSO"
            elif sel > 0:
                note = "✓ Incluso"
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
    
    # Statistiche di bilanciamento EXPOSURE
    print(f"\n🎯 BILANCIAMENTO EXPOSURE (threshold SASA={sasa_threshold}):")
    print(f"  Exposed: {total_exposed} ({total_exposed/(total_exposed+total_buried)*100:.1f}%)")
    print(f"  Buried:  {total_buried} ({total_buried/(total_exposed+total_buried)*100:.1f}%)")
    
    exposure_imbalance = abs(total_exposed - total_buried)
    if exposure_imbalance <= 2:
        print("  ✅ Bilanciamento EXPOSURE ottimo!")
    elif exposure_imbalance <= 10:
        print("  ⚠️  Bilanciamento EXPOSURE accettabile")
    else:
        print("  ❌ Bilanciamento EXPOSURE sbilanciato")
    
    # Statistiche di bilanciamento AA (per gruppo)
    for exposure_type, samples in [('EXPOSED', selected_per_aa_exposed), 
                                    ('BURIED', selected_per_aa_buried)]:
        if samples:
            print(f"\n🎯 BILANCIAMENTO AA ({exposure_type}, esclusi AA rari):")
            print(f"  Min campioni: {min(samples)}")
            print(f"  Max campioni: {max(samples)}")
            print(f"  Media: {np.mean(samples):.2f}")
            print(f"  Std dev: {np.std(samples):.2f}")
            print(f"  Range: {max(samples) - min(samples)}")
            
            aa_imbalance = max(samples) - min(samples)
            if aa_imbalance <= 2:
                print(f"  ✅ Bilanciamento AA ottimo in {exposure_type}!")
            elif aa_imbalance <= 5:
                print(f"  ⚠️  Bilanciamento AA accettabile in {exposure_type}")
            else:
                print(f"  ❌ Bilanciamento AA sbilanciato in {exposure_type}")
    
    # Analisi distribuzione SASA complessiva nel test set
    print(f"\n📊 DISTRIBUZIONE SASA COMPLESSIVA NEL TEST SET:")
    all_sasa = []
    for window in selected:
        residues = window.get('residues', [])
        if len(residues) >= 5:
            sasa = residues[2].get('ACC_norm')
            if sasa is not None:
                all_sasa.append(float(sasa))
    
    if all_sasa:
        print(f"  Media SASA: {np.mean(all_sasa):.3f}")
        print(f"  Std SASA: {np.std(all_sasa):.3f}")
        print(f"  Min SASA: {np.min(all_sasa):.3f}")
        print(f"  Max SASA: {np.max(all_sasa):.3f}")
        print(f"  Mediana SASA: {np.median(all_sasa):.3f}")
        
        # Verifica distribuzione rispetto al threshold
        n_above_threshold = sum(1 for s in all_sasa if s >= sasa_threshold)
        n_below_threshold = len(all_sasa) - n_above_threshold
        print(f"  Campioni SASA >= {sasa_threshold}: {n_above_threshold} ({n_above_threshold/len(all_sasa)*100:.1f}%)")
        print(f"  Campioni SASA < {sasa_threshold}: {n_below_threshold} ({n_below_threshold/len(all_sasa)*100:.1f}%)")
    
    # Analisi distribuzione SS nel test set
    print(f"\n📊 DISTRIBUZIONE SECONDARY STRUCTURE NEL TEST SET:")
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
    """Salva le finestre in formato CSV"""
    df = pd.DataFrame(windows)
    df.to_csv(output_path, index=False)
    print(f"✓ CSV salvato: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Crea un test set bilanciato per AA e EXPOSURE (usando SASA threshold) preservando SS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempio d'uso:
  python create_balanced_testset.py \\
    --input windows_all.json \\
    --output windows_testset_balanced.json \\
    --n_samples 300 \\
    --sasa_threshold 0.1199 \\
    --remove_from_original \\
    --seed 42

Note:
  - Bilancia su 50% exposed / 50% buried (usando SASA threshold)
  - All'interno di ogni gruppo, distribuzione uniforme degli AA
  - Mantiene la distribuzione naturale di SASA e SS per ogni (AA, exposure)
  - ESCLUDE automaticamente AA rari: P, W, C, M, R
  - Gli AA esclusi rimangono nel training set
  - Preserva la feature SS (Secondary Structure) per ogni residuo
        """
    )
    ap.add_argument("--input", required=True, 
                    help="File JSON con le finestre")
    ap.add_argument("--output", required=True,
                    help="File JSON di output per il test set bilanciato")
    ap.add_argument("--output_csv", 
                    help="File CSV di output (default: stesso nome del JSON con .csv)")
    ap.add_argument("--n_samples", type=int, default=300,
                    help="Numero totale di campioni da selezionare (default: 300)")
    ap.add_argument("--sasa_threshold", type=float, default=0.1199,
                    help="Soglia SASA per definire exposed (>= threshold) vs buried (< threshold)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed per riproducibilità (default: 42)")
    ap.add_argument("--remove_from_original", action="store_true",
                    help="Rimuovi i campioni selezionati dal file originale")
    
    args = ap.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Determina il path del CSV
    if args.output_csv:
        output_csv_path = Path(args.output_csv)
    else:
        output_csv_path = output_path.with_suffix('.csv')
    
    if not input_path.exists():
        raise SystemExit(f"✗ File non trovato: {input_path}")
    
    print("="*140)
    print("CARICAMENTO DATI - BILANCIAMENTO SU AA + EXPOSURE (SASA THRESHOLD) + SS")
    print("="*140)
    print(f"Input: {input_path}")
    print(f"Output JSON: {output_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"N samples richiesti: {args.n_samples}")
    print(f"SASA threshold: {args.sasa_threshold}")
    print(f"Seed: {args.seed}")
    
    # Carica finestre
    windows = load_windows(input_path)
    print(f"Finestre totali: {len(windows)}")
    
    # Raggruppa per (AA, exposure)
    print("\nRaggruppamento per (amminoacido, exposure) con analisi SS...")
    grouped, sasa_values, ss_distributions = group_by_aa_and_exposure(
        windows, 
        sasa_threshold=args.sasa_threshold,
        debug=True
    )
    
    print(f"\nCombinazioni (AA, exposure) trovate: {len(grouped)}")
    
    # Analisi disponibilità per exposure
    exposed_total = sum(len(v) for k, v in grouped.items() if k[1] == 'exposed')
    buried_total = sum(len(v) for k, v in grouped.items() if k[1] == 'buried')
    
    print(f"\nDistribuzione nel dataset originale:")
    print(f"  EXPOSED (SASA >= {args.sasa_threshold}): {exposed_total} finestre ({exposed_total/(exposed_total+buried_total)*100:.1f}%)")
    print(f"  BURIED (SASA < {args.sasa_threshold}): {buried_total} finestre ({buried_total/(exposed_total+buried_total)*100:.1f}%)")
    
    # Seleziona campioni bilanciati
    print("\n" + "="*140)
    print("SELEZIONE BILANCIATA")
    print("="*140)
    
    selected, stats = select_balanced_samples(
        grouped, sasa_values, ss_distributions,
        args.n_samples, args.seed
    )
    
    # Statistiche
    print_statistics(stats, selected, args.sasa_threshold)
    
    # Salva output JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Test set bilanciato salvato: {output_path}")
    print(f"   Campioni selezionati: {len(selected)}")
    
    # Salva output CSV
    save_to_csv(selected, output_csv_path)
    
    # Rimuovi dal file originale se richiesto
    if args.remove_from_original:
        print("\nRimozione campioni dal file originale...")
        
        # Crea set di ID selezionati
        selected_ids = {w.get('window_id') for w in selected if 'window_id' in w}
        
        # Mantieni finestre non selezionate
        remaining = [w for w in windows if w.get('window_id') not in selected_ids]
        
        with input_path.open('w', encoding='utf-8') as f:
            json.dump(remaining, f, ensure_ascii=False, indent=2)
        
        print(f"✅ File originale aggiornato: {input_path}")
        print(f"   Finestre rimanenti: {len(remaining)}")
        print(f"   Finestre rimosse: {len(windows) - len(remaining)}")
        print(f"\n💡 Gli AA esclusi ({', '.join(sorted(EXCLUDED_AA))}) sono rimasti TUTTI nel training set")


if __name__ == "__main__":
    main()