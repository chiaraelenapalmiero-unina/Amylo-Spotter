#!/usr/bin/env python3
"""
Script per convertire cluster da indici GLOBALI (0-97) a LOCALI (0-N per ogni AA).
Versione aggiornata per dataset con SASA continua e SS (Secondary Structure).

Uso:
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

# rotamer_library.py vive in ../common (dipendenza condivisa con model/)
sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from rotamer_library import load_rotamer_library, AA20

def convert_global_to_local(aa_ids, cluster_global, rotamer_lib):
    """
    Converte cluster da indici globali a locali.
    
    Args:
        aa_ids: array [N, L] con indici aminoacidi
        cluster_global: array [N, L] con indici cluster GLOBALI (0-97)
        rotamer_lib: istanza di RotamerLibrary
    
    Returns:
        cluster_local: array [N, L] con indici cluster LOCALI (0-n per ogni AA)
        conversion_stats: dict con statistiche
    """
    cluster_local = cluster_global.copy()
    
    # Statistiche conversione
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
            
            # Salta MASK tokens (AA >= 20)
            if aa_idx >= 20:
                stats['mask'] += 1
                continue
            
            # Salta cluster MASK (assumiamo che cluster molto alti siano mask)
            if cl_global >= 15:  # Threshold ragionevole
                stats['mask'] += 1
                continue
            
            # Cerca se questo cluster globale corrisponde a questo AA
            # nella libreria di rotameri
            found = False
            
            for local_idx in range(rotamer_lib.get_n_clusters(aa_idx)):
                global_idx = rotamer_lib.pair_to_global.get((aa_idx, local_idx), -1)
                
                if global_idx == cl_global:
                    # Trovato! Converti
                    cluster_local[i, j] = local_idx
                    stats['converted'] += 1
                    found = True
                    break
            
            if not found:
                # Non trovato - questo cluster globale non appartiene a questo AA
                # Prova a mapparlo comunque prendendo il resto
                n_clusters = rotamer_lib.get_n_clusters(aa_idx)
                if n_clusters > 0:
                    # Fallback: mappa con modulo
                    local_fallback = cl_global % n_clusters
                    cluster_local[i, j] = local_fallback
                    stats['invalid'] += 1
                else:
                    stats['invalid'] += 1
    
    return cluster_local, stats


def verify_conversion(aa_ids, cluster_local, rotamer_lib):
    """
    Verifica che dopo la conversione tutte le coppie siano valide.
    """
    N, L = aa_ids.shape
    invalid_count = 0
    valid_count = 0
    
    for i in range(N):
        for j in range(L):
            aa_idx = int(aa_ids[i, j])
            cl_idx = int(cluster_local[i, j])
            
            # Salta MASK
            if aa_idx >= 20:
                continue
            
            valid_clusters = rotamer_lib.get_valid_clusters(aa_idx)
            if cl_idx in valid_clusters:
                valid_count += 1
            else:
                invalid_count += 1
                if invalid_count <= 5:  # Stampa primi 5
                    aa_name = AA20[aa_idx] if aa_idx < 20 else "?"
                    print(f"  ⚠️  Still invalid: AA={aa_idx} ({aa_name}), cluster={cl_idx}, "
                          f"valid={valid_clusters}")
    
    return valid_count, invalid_count


def analyze_ss_distribution(ss_ids):
    """
    Analizza la distribuzione della struttura secondaria nel dataset.
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
    parser.add_argument("--input", required=True, help="Input NPZ file")
    parser.add_argument("--output", required=True, help="Output NPZ file")
    parser.add_argument("--rotamer_csv", default="rotamer_localmax_maxdens.csv", 
                       help="Rotamer library CSV")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify conversion after completion")
    parser.add_argument("--analyze_ss", action="store_true",
                       help="Analyze Secondary Structure distribution")
    
    args = parser.parse_args()
    
    # Carica rotamer library
    print("="*60)
    print("LOADING ROTAMER LIBRARY")
    print("="*60)
    lib = load_rotamer_library(args.rotamer_csv)
    
    # Carica dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    print(f"Input file: {args.input}")
    
    data = np.load(args.input, allow_pickle=True)  # 🆕 allow_pickle per pdb_id
    aa_ids = data["aa_ids"]
    cluster_global = data["cluster_idx"]
    
    print(f"Shape: {aa_ids.shape}")
    print(f"AA range: {aa_ids.min()} - {aa_ids.max()}")
    print(f"Cluster range (global): {cluster_global.min()} - {cluster_global.max()}")
    
    # Controlla se c'è SASA
    if "sasa" in data:
        print(f"SASA found - range: {data['sasa'].min():.4f} - {data['sasa'].max():.4f}")
        print(f"SASA mean: {data['sasa'].mean():.4f}")
        print(f"SASA std: {data['sasa'].std():.4f}")
    elif "exposure" in data:
        print(f"Exposure found (binary) - using legacy format")
    
    # Controlla se c'è SS
    has_ss = "ss" in data
    if has_ss:
        ss_ids = data["ss"]
        print(f"SS (Secondary Structure) found - range: {ss_ids.min()} - {ss_ids.max()}")
        print(f"SS shape: {ss_ids.shape}")
        
        if args.analyze_ss:
            analyze_ss_distribution(ss_ids)
    else:
        print("⚠️  No Secondary Structure (SS) data found in dataset")
    
    # Controlla se c'è pos_idx
    if "pos_idx" in data:
        pos_idx = data["pos_idx"]
        print(f"Position indices found - range: {pos_idx.min()} - {pos_idx.max()}")
    
    # 🆕 Controlla protein_class_id
    if "protein_class_id" in data:
        protein_class_ids = data["protein_class_id"]
        print(f"Protein Class IDs found - range: {protein_class_ids.min()} - {protein_class_ids.max()}")
        unique_classes = np.unique(protein_class_ids)
        print(f"  Unique protein classes: {len(unique_classes)}")
    
    # 🆕 Controlla pdb_id
    if "pdb_id" in data:
        pdb_ids = data["pdb_id"]
        print(f"PDB IDs found - total: {len(pdb_ids)}")
        unique_pdbs = len(set(pdb_ids))
        print(f"  Unique PDB IDs: {unique_pdbs}")
    
    # Converti
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
    
    # Verifica
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
                print("\n✅ ALL PAIRS ARE NOW VALID!")
            else:
                print(f"\n⚠️  Still {invalid} invalid pairs. Check warnings above.")
        else:
            print("No non-masked pairs to validate.")
    
    # Salva
    print("\n" + "="*60)
    print("SAVING CONVERTED DATASET")
    print("="*60)
    
    # Copia tutti gli altri campi (incluso SASA e SS se presenti)
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
    
    # Mostra campi salvati
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
        print("\n✅ Secondary Structure (SS) data preserved in output")
    else:
        print("\n⚠️  No SS data in input - consider regenerating dataset with SS features")


if __name__ == "__main__":
    main()