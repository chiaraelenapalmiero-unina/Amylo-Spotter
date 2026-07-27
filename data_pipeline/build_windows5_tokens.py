#python build_windows5_tokens_with_pdb_protein.py --input /mnt/user-data/uploads/windows_all.csv --output windows5_tokens.npz legge solo json
import argparse, os, glob, json, numpy as np

AA3_to_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}
AA1_order = list("ACDEFGHIKLMNPQRSTVWY")  # Alfabetico come rotamer_library!
AA1_to_id = {aa:i for i,aa in enumerate(AA1_order)}    # 0..19
AA3_to_id = {k: AA1_to_id[v] for k,v in AA3_to_1.items()}

POS_LIST = [-2, -1, 0, 1, 2]
POS_MAP  = {-2:0, -1:1, 0:2, 1:3, 2:4}

# 🆕 MAPPING SECONDARY STRUCTURE
# Mappa caratteri SS comuni (verrà usata come fallback)
SS_COMMON_MAP = {
    'H': 'H',  # Helix
    'G': 'G',  # 3-10 helix
    'I': 'I',  # Pi helix
    'E': 'E',  # Beta strand
    'B': 'B',  # Beta bridge
    'T': 'T',  # Turn
    'S': 'S',  # Bend
    'C': 'C',  # Coil
    '-': 'C',  # Unknown → Coil
    ' ': 'C',  # Empty → Coil
    None: 'C'  # Missing → Coil
}


def parse_args():
    ap = argparse.ArgumentParser(description="Build token arrays from 5-residue windows JSONs with SASA, SS, PDB_ID, and PROTEIN_CLASS.")
    ap.add_argument("--input", required=True, help="File .json o cartella con file .json (ricorsivo).")
    ap.add_argument("--output", required=True, help="Percorso del file .npz in output.")
    ap.add_argument("--include_meta", action="store_true", help="Salva anche meta (pdb, face, strand_start).")
    return ap.parse_args()

def iter_json_files(root):
    """Itera sui file JSON. Se root è un file, ritorna solo quello."""
    if os.path.isfile(root) and root.endswith('.json'):
        yield root
        return
    
    pats = ["**/*.json", "*.json"]
    seen = set()
    for pat in pats:
        for p in glob.glob(os.path.join(root, pat), recursive=True):
            if p not in seen:
                seen.add(p)
                yield p

def extract_windows5_from_json(path):
    """Legge un file JSON che contiene una lista di finestre."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return []

    wins = []
    if isinstance(data, list):
        for w in data:
            if not isinstance(w, dict):
                continue
            residues = w.get("residues", [])
            if isinstance(residues, list) and len(residues) == 5:
                wins.append(w)
    return wins

def parse_sasa_value(sasa_val):
    """Converte il valore SASA/ACC_norm in float, gestendo valori mancanti."""
    if sasa_val is None:
        return 0.0
    try:
        val = float(sasa_val)
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return 0.0

def parse_ss_value(ss_val):
    """
    🆕 Normalizza il valore SS (stringa DSSP) mantenendo il carattere originale.
    
    Args:
        ss_val: carattere SS ('H', 'E', 'C', etc.) o None
    
    Returns:
        str: carattere SS normalizzato
    """
    if ss_val is None or ss_val == '':
        return 'C'  # Default: Coil
    
    # Prendi solo il primo carattere se è una stringa
    ss_char = str(ss_val).strip().upper()[0] if ss_val else 'C'
    
    # Normalizza caratteri non standard
    return SS_COMMON_MAP.get(ss_char, 'C')

def build_ss_mapping(ss_chars):
    """
    🆕 Costruisce automaticamente il mapping SS dai caratteri trovati nei dati.
    
    Args:
        ss_chars: set di caratteri SS unici trovati
    
    Returns:
        tuple: (ss_to_idx dict, unique_ss_list)
    """
    # Rimuovi None e ordina alfabeticamente per consistenza
    unique_ss = sorted([s for s in ss_chars if s is not None and s != ''])
    
    # Crea mapping: carattere -> indice
    ss_to_idx = {ss_char: idx for idx, ss_char in enumerate(unique_ss)}
    
    return ss_to_idx, unique_ss

def build_protein_class_mapping(protein_classes):
    """
    🆕 Costruisce mapping automatico per le classi di proteine.
    
    Args:
        protein_classes: set di nomi di classi di proteine
    
    Returns:
        tuple: (class_to_idx dict, unique_classes_list)
    """
    # Ordina alfabeticamente per consistenza
    unique_classes = sorted([c for c in protein_classes if c is not None and c != ''])
    
    # Crea mapping: nome classe -> indice
    class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
    
    return class_to_idx, unique_classes

def main():
    args = parse_args()
    root = args.input
    out_npz = args.output
    out_dir = os.path.dirname(out_npz) or "."
    os.makedirs(out_dir, exist_ok=True)

    all_windows = []
    for jf in iter_json_files(root):
        w = extract_windows5_from_json(jf)
        if w:
            all_windows.extend(w)

    if not all_windows:
        raise SystemExit("Nessuna finestra da 5 residui trovata. Controlla la cartella di input.")

    # 🆕 FASE 1: Scansione per trovare tutte le classi SS uniche E le protein classes
    print("="*60)
    print("PHASE 1: Scanning for unique classes...")
    print("="*60)
    
    unique_ss_chars = set()
    unique_protein_classes = set()
    ss_missing_count = 0
    protein_class_missing_count = 0
    pdb_missing_count = 0
    total_residues = 0
    
    for w in all_windows:
        residues = w["residues"]
        
        # 🆕 Raccogli protein_class e pdb_id
        protein_class = w.get("protein_class", None)
        pdb_id = w.get("pdb", None)
        
        if protein_class:
            unique_protein_classes.add(protein_class)
        else:
            protein_class_missing_count += 1
            
        if not pdb_id:
            pdb_missing_count += 1
        
        for r in residues:
            total_residues += 1
            ss_val = r.get("ss", r.get("SS", r.get("secondary_structure", r.get("dssp", None))))
            ss_char = parse_ss_value(ss_val)
            
            if ss_val is None:
                ss_missing_count += 1
            
            unique_ss_chars.add(ss_char)
    
    # 🆕 Costruisci mapping automatici
    ss_to_idx, unique_ss_list = build_ss_mapping(unique_ss_chars)
    n_ss_classes = len(ss_to_idx)
    
    protein_class_to_idx, unique_protein_class_list = build_protein_class_mapping(unique_protein_classes)
    n_protein_classes = len(protein_class_to_idx)
    
    print(f"Found {n_ss_classes} unique SS classes: {unique_ss_list}")
    print(f"Found {n_protein_classes} unique protein classes: {unique_protein_class_list}")
    print(f"\nTotal windows: {len(all_windows)}")
    print(f"Total residues scanned: {total_residues}")
    print(f"Missing SS values: {ss_missing_count} ({ss_missing_count/total_residues*100:.1f}%)")
    print(f"Missing protein_class values: {protein_class_missing_count} ({protein_class_missing_count/len(all_windows)*100:.1f}%)")
    print(f"Missing pdb_id values: {pdb_missing_count} ({pdb_missing_count/len(all_windows)*100:.1f}%)")
    
    print(f"\nSS Mapping:")
    for ss_char, idx in sorted(ss_to_idx.items()):
        print(f"  '{ss_char}' -> {idx}")
    
    print(f"\nProtein Class Mapping:")
    for class_name, idx in sorted(protein_class_to_idx.items()):
        print(f"  '{class_name}' -> {idx}")
    
    # 🆕 FASE 2: Build arrays con i mapping costruiti
    print("\n" + "="*60)
    print("PHASE 2: Building arrays...")
    print("="*60)

    aa_list = []
    cl_list = []
    sasa_list = []
    ss_list = []  # 🆕 Lista per Secondary Structure
    protein_class_list = []  # 🆕 Lista per Protein Class ID
    pdb_id_list = []  # 🆕 Lista per PDB ID
    pos_list = []
    meta_list = []

    for w in all_windows:
        residues = w["residues"]
        pos_vals = [r.get("win_pos", None) for r in residues]
        if not all(p in POS_MAP for p in pos_vals):
            pos_vals = POS_LIST

        # 🆕 Estrai protein_class e pdb_id dalla finestra
        protein_class = w.get("protein_class", None)
        pdb_id = w.get("pdb", "")
        
        # Converti protein_class in ID numerico (-1 se mancante)
        if protein_class and protein_class in protein_class_to_idx:
            protein_class_id = protein_class_to_idx[protein_class]
        else:
            protein_class_id = -1
        
        aa_ids = []
        cl_ids = []
        sasa_vals = []
        ss_vals = []  # 🆕 Valori SS per questa finestra
        
        for r in residues:
            aa3 = str(r.get("AA","")).upper()
            aa_id = AA3_to_id.get(aa3, 0)
            clid = r.get("cluster_id", None)
            if clid is None:
                clid = -1
            
            # SASA
            sasa_val = r.get("ACC_norm", r.get("sasa", None))
            sasa_val = parse_sasa_value(sasa_val)
            
            # 🆕 SECONDARY STRUCTURE
            ss_val = r.get("ss", r.get("SS", r.get("secondary_structure", r.get("dssp", None))))
            ss_char = parse_ss_value(ss_val)
            ss_id = ss_to_idx[ss_char]  # Converti a indice
            
            aa_ids.append(aa_id)
            cl_ids.append(int(clid))
            sasa_vals.append(sasa_val)
            ss_vals.append(ss_id)  # 🆕 Aggiungi SS

        aa_list.append(aa_ids)
        cl_list.append(cl_ids)
        sasa_list.append(sasa_vals)
        ss_list.append(ss_vals)  # 🆕
        protein_class_list.append(protein_class_id)  # 🆕
        pdb_id_list.append(pdb_id)  # 🆕
        pos_list.append(pos_vals)

        if args.include_meta:
            meta_list.append([w.get("pdb",""), w.get("face",""), w.get("strand_start", None)])

    aa_arr = np.array(aa_list, dtype=np.int16)
    cl_arr = np.array(cl_list, dtype=np.int64)
    sasa_arr = np.array(sasa_list, dtype=np.float32)
    ss_arr = np.array(ss_list, dtype=np.int8)  # 🆕 Array SS
    protein_class_arr = np.array(protein_class_list, dtype=np.int16)  # 🆕 Array Protein Class
    pdb_id_arr = np.array(pdb_id_list, dtype=object)  # 🆕 Array PDB ID (stringhe)
    pos_arr = np.array(pos_list, dtype=np.int16)

    # Remap cluster ids
    unique_clusters = np.unique(cl_arr)
    cluster_to_idx = {int(c): int(i) for i, c in enumerate(unique_clusters)}
    cl_idx = np.vectorize(lambda x: cluster_to_idx[int(x)])(cl_arr).astype(np.int32)

    # Remap positions
    pos_idx = np.vectorize(lambda x: POS_MAP[int(x)])(pos_arr).astype(np.int8)

    # 🆕 Save NPZ con SS, Protein Class e PDB ID
    np.savez_compressed(
        out_npz,
        aa_ids=aa_arr,
        cluster_idx=cl_idx,
        sasa=sasa_arr,
        ss=ss_arr,                      # 🆕 Campo Secondary Structure
        protein_class_id=protein_class_arr,  # 🆕 Campo Protein Class ID
        pdb_id=pdb_id_arr,             # 🆕 Campo PDB ID
        pos_idx=pos_idx,
    )

    # Save mappings
    with open(os.path.join(out_dir, "cluster_id_mapping.json"), "w") as f:
        json.dump({"cluster_to_idx": cluster_to_idx}, f, indent=2)
    with open(os.path.join(out_dir, "pos_mapping.json"), "w") as f:
        json.dump({str(k): v for k, v in POS_MAP.items()}, f, indent=2)
    with open(os.path.join(out_dir, "aa_id_mapping.json"), "w") as f:
        json.dump({
            "AA1_order": AA1_order,
            "note": "IDs run from 0..19; aminoacidi in ordine alfabetico."
        }, f, indent=2)
    
    # 🆕 Save SS mapping
    idx_to_ss = {v: k for k, v in ss_to_idx.items()}
    with open(os.path.join(out_dir, "ss_mapping.json"), "w") as f:
        json.dump({
            "type": "auto-detected",
            "n_classes": n_ss_classes,
            "ss_to_idx": ss_to_idx,
            "idx_to_ss": idx_to_ss,
            "unique_classes": unique_ss_list,
            "note": f"Auto-detected {n_ss_classes} SS classes from data"
        }, f, indent=2)
    
    # 🆕 Save Protein Class mapping
    idx_to_protein_class = {v: k for k, v in protein_class_to_idx.items()}
    with open(os.path.join(out_dir, "protein_class_mapping.json"), "w") as f:
        json.dump({
            "type": "auto-detected",
            "n_classes": n_protein_classes,
            "protein_class_to_idx": protein_class_to_idx,
            "idx_to_protein_class": idx_to_protein_class,
            "unique_classes": unique_protein_class_list,
            "note": f"Auto-detected {n_protein_classes} protein classes from data"
        }, f, indent=2)

    # Optional meta
    if args.include_meta:
        np.savez_compressed(os.path.join(out_dir, "windows5_meta.npz"), 
                          meta=np.array(meta_list, dtype=object))

    # Statistics
    print(f"\n{'='*60}")
    print(f"Salvato: {out_npz}")
    print(f"{'='*60}")
    print(f"Finestre N = {aa_arr.shape[0]}, cluster K = {len(unique_clusters)}")
    
    print(f"\n📊 SASA statistics:")
    print(f"  Min:    {sasa_arr.min():.4f}")
    print(f"  Max:    {sasa_arr.max():.4f}")
    print(f"  Mean:   {sasa_arr.mean():.4f}")
    print(f"  Median: {np.median(sasa_arr):.4f}")
    
    # 🆕 SS statistics
    print(f"\n🧬 SECONDARY STRUCTURE statistics:")
    print(f"  Type: auto-detected")
    print(f"  Classes: {n_ss_classes}")
    print(f"  Missing SS values: {ss_missing_count}/{total_residues} ({ss_missing_count/total_residues*100:.1f}%)")
    
    unique_ss, counts_ss = np.unique(ss_arr, return_counts=True)
    print(f"  Distribution:")
    for ss_idx, count in zip(unique_ss, counts_ss):
        ss_char = idx_to_ss.get(ss_idx, '?')
        percentage = count / ss_arr.size * 100
        print(f"    Class {ss_idx} ('{ss_char}'): {count:6d} ({percentage:5.1f}%)")
    
    # 🆕 Protein Class statistics
    print(f"\n🧪 PROTEIN CLASS statistics:")
    print(f"  Type: auto-detected")
    print(f"  Classes: {n_protein_classes}")
    print(f"  Missing protein_class values: {protein_class_missing_count}/{len(all_windows)} ({protein_class_missing_count/len(all_windows)*100:.1f}%)")
    
    unique_pc, counts_pc = np.unique(protein_class_arr, return_counts=True)
    print(f"  Distribution:")
    for pc_idx, count in zip(unique_pc, counts_pc):
        if pc_idx == -1:
            pc_name = "MISSING"
        else:
            pc_name = idx_to_protein_class.get(pc_idx, 'UNKNOWN')
        percentage = count / protein_class_arr.size * 100
        print(f"    Class {pc_idx:3d} ('{pc_name}'): {count:6d} ({percentage:5.1f}%)")
    
    # 🆕 PDB ID statistics
    print(f"\n🔬 PDB ID statistics:")
    unique_pdbs = set(pdb_id_arr)
    print(f"  Unique PDB IDs: {len(unique_pdbs)}")
    print(f"  Missing PDB IDs: {pdb_missing_count}/{len(all_windows)} ({pdb_missing_count/len(all_windows)*100:.1f}%)")
    
    print(f"\n✅ Mapping salvati in: {out_dir}")
    print("="*60)

if __name__ == "__main__":
    main()