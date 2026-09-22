#python build_windows5_tokens_with_pdb_protein.py --input /mnt/user-data/uploads/windows_all.csv --output windows5_tokens.npz  (reads only .json files, despite the --input example above)
"""
Build token arrays (amino acid, cluster, SASA, secondary structure,
protein class, PDB ID, window position) from a set of 5-residue-window
JSON files, and save them as a single compressed .npz file plus a set of
JSON mapping files describing how each categorical value was encoded.

Usage:
    python build_windows5_tokens_with_pdb_protein.py \
        --input /path/to/windows_or_folder \
        --output windows5_tokens.npz \
        [--include_meta]
"""
import argparse, os, glob, json, numpy as np

AA3_to_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}
AA1_order = list("ACDEFGHIKLMNPQRSTVWY")  # Alphabetical order, same as rotamer_library!
AA1_to_id = {aa:i for i,aa in enumerate(AA1_order)}    # 0..19
AA3_to_id = {k: AA1_to_id[v] for k,v in AA3_to_1.items()}

POS_LIST = [-2, -1, 0, 1, 2]
POS_MAP  = {-2:0, -1:1, 0:2, 1:3, 2:4}

# SECONDARY STRUCTURE MAPPING
# Maps common DSSP SS characters to a normalized set (used as a fallback
# whenever a character outside this table is encountered).
SS_COMMON_MAP = {
    'H': 'H',  # Helix
    'G': 'G',  # 3-10 helix
    'I': 'I',  # Pi helix
    'E': 'E',  # Beta strand
    'B': 'B',  # Beta bridge
    'T': 'T',  # Turn
    'S': 'S',  # Bend
    'C': 'C',  # Coil
    '-': 'C',  # Unknown -> Coil
    ' ': 'C',  # Empty -> Coil
    None: 'C'  # Missing -> Coil
}


def parse_args():
    """
    Defines and parses the command-line arguments for this script.

    Command-line arguments:
        --input        Path to a single .json window file, or to a folder
                        that will be searched recursively for .json files.
                        Required.
        --output       Path to the .npz file that will be written with the
                        encoded token arrays. Required.
        --include_meta Optional flag. If set, also saves a companion
                        "<output_dir>/windows5_meta.npz" file containing,
                        for each window, its PDB id, face, and strand start
                        position (useful for tracing a token back to its
                        source window).

    Returns:
        argparse.Namespace with attributes `input`, `output`, `include_meta`.
    """
    ap = argparse.ArgumentParser(description="Build token arrays from 5-residue windows JSONs with SASA, SS, PDB_ID, and PROTEIN_CLASS.")
    ap.add_argument("--input", required=True, help="A single .json window file, or a folder containing .json window files (searched recursively).")
    ap.add_argument("--output", required=True, help="Path to the output .npz file that will hold the encoded token arrays.")
    ap.add_argument("--include_meta", action="store_true", help="Also save per-window metadata (pdb, face, strand_start) to a companion windows5_meta.npz file.")
    return ap.parse_args()

def iter_json_files(root):
    """
    Iterates over JSON files under `root`.

    Args:
        root: Path to a single .json file, or to a directory to search
            recursively for .json files.

    Yields:
        Paths to each .json file found. If `root` is itself a .json file,
        yields only that path.
    """
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
    """
    Reads a JSON file expected to contain a list of window dicts, and
    returns only the ones that describe a 5-residue window.

    Args:
        path: Path to the JSON file to read.

    Returns:
        List of window dicts (each with a "residues" list of length 5).
        Returns an empty list if the file cannot be read/parsed, or
        contains no valid 5-residue windows.
    """
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
    """
    Converts a SASA/ACC_norm value to a float, clamped to [0, 1], handling
    missing or malformed values.

    Args:
        sasa_val: Raw SASA/ACC_norm value from the input JSON (may be a
            number, string, or None).

    Returns:
        float in the range [0.0, 1.0]. Returns 0.0 if the value is missing
        or cannot be converted to a number.
    """
    if sasa_val is None:
        return 0.0
    try:
        val = float(sasa_val)
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return 0.0

def parse_ss_value(ss_val):
    """
    Normalizes a secondary-structure (SS) value coming from DSSP, keeping
    the original character where possible.

    Args:
        ss_val: SS character ('H', 'E', 'C', etc.) or None.

    Returns:
        str: A single normalized SS character (one of the keys/values in
            SS_COMMON_MAP; defaults to 'C' for missing/unrecognized input).
    """
    if ss_val is None or ss_val == '':
        return 'C'  # Default: Coil
    
    # Take only the first character if it's a string
    ss_char = str(ss_val).strip().upper()[0] if ss_val else 'C'
    
    # Normalize non-standard characters
    return SS_COMMON_MAP.get(ss_char, 'C')

def build_ss_mapping(ss_chars):
    """
    Automatically builds the SS -> integer-index mapping from the set of
    SS characters actually observed in the data.

    Args:
        ss_chars: Set of unique SS characters found in the dataset.

    Returns:
        tuple:
            ss_to_idx (dict): Maps each SS character to an integer index.
            unique_ss_list (list): Sorted list of the SS characters used,
                in the same order as their assigned indices.
    """
    # Drop None and sort alphabetically for a consistent, reproducible mapping
    unique_ss = sorted([s for s in ss_chars if s is not None and s != ''])
    
    # Build mapping: character -> index
    ss_to_idx = {ss_char: idx for idx, ss_char in enumerate(unique_ss)}
    
    return ss_to_idx, unique_ss

def build_protein_class_mapping(protein_classes):
    """
    Automatically builds the protein-class -> integer-index mapping from
    the set of protein class names actually observed in the data.

    Args:
        protein_classes: Set of unique protein class names found in the
            dataset.

    Returns:
        tuple:
            class_to_idx (dict): Maps each protein class name to an
                integer index.
            unique_classes_list (list): Sorted list of protein class
                names, in the same order as their assigned indices.
    """
    # Sort alphabetically for a consistent, reproducible mapping
    unique_classes = sorted([c for c in protein_classes if c is not None and c != ''])
    
    # Build mapping: class name -> index
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
        raise SystemExit("No 5-residue windows found. Check the input path.")

    # PHASE 1: Scan to find all unique SS classes AND protein classes
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
        
        # Collect protein_class and pdb_id
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
    
    # Build automatic mappings
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
    
    # PHASE 2: Build arrays using the mappings just built
    print("\n" + "="*60)
    print("PHASE 2: Building arrays...")
    print("="*60)

    aa_list = []
    cl_list = []
    sasa_list = []
    ss_list = []  # List for Secondary Structure
    protein_class_list = []  # List for Protein Class ID
    pdb_id_list = []  # List for PDB ID
    pos_list = []
    meta_list = []

    for w in all_windows:
        residues = w["residues"]
        pos_vals = [r.get("win_pos", None) for r in residues]
        if not all(p in POS_MAP for p in pos_vals):
            pos_vals = POS_LIST

        # Extract protein_class and pdb_id from the window
        protein_class = w.get("protein_class", None)
        pdb_id = w.get("pdb", "")
        
        # Convert protein_class to a numeric ID (-1 if missing)
        if protein_class and protein_class in protein_class_to_idx:
            protein_class_id = protein_class_to_idx[protein_class]
        else:
            protein_class_id = -1
        
        aa_ids = []
        cl_ids = []
        sasa_vals = []
        ss_vals = []  # SS values for this window
        
        for r in residues:
            aa3 = str(r.get("AA","")).upper()
            aa_id = AA3_to_id.get(aa3, 0)
            clid = r.get("cluster_id", None)
            if clid is None:
                clid = -1
            
            # SASA
            sasa_val = r.get("ACC_norm", r.get("sasa", None))
            sasa_val = parse_sasa_value(sasa_val)
            
            # SECONDARY STRUCTURE
            ss_val = r.get("ss", r.get("SS", r.get("secondary_structure", r.get("dssp", None))))
            ss_char = parse_ss_value(ss_val)
            ss_id = ss_to_idx[ss_char]  # Convert to index
            
            aa_ids.append(aa_id)
            cl_ids.append(int(clid))
            sasa_vals.append(sasa_val)
            ss_vals.append(ss_id)  # Append SS

        aa_list.append(aa_ids)
        cl_list.append(cl_ids)
        sasa_list.append(sasa_vals)
        ss_list.append(ss_vals)
        protein_class_list.append(protein_class_id)
        pdb_id_list.append(pdb_id)
        pos_list.append(pos_vals)

        if args.include_meta:
            meta_list.append([w.get("pdb",""), w.get("face",""), w.get("strand_start", None)])

    aa_arr = np.array(aa_list, dtype=np.int16)
    cl_arr = np.array(cl_list, dtype=np.int64)
    sasa_arr = np.array(sasa_list, dtype=np.float32)
    ss_arr = np.array(ss_list, dtype=np.int8)  # SS array
    protein_class_arr = np.array(protein_class_list, dtype=np.int16)  # Protein Class array
    pdb_id_arr = np.array(pdb_id_list, dtype=object)  # PDB ID array (strings)
    pos_arr = np.array(pos_list, dtype=np.int16)

    # Remap cluster ids
    unique_clusters = np.unique(cl_arr)
    cluster_to_idx = {int(c): int(i) for i, c in enumerate(unique_clusters)}
    cl_idx = np.vectorize(lambda x: cluster_to_idx[int(x)])(cl_arr).astype(np.int32)

    # Remap positions
    pos_idx = np.vectorize(lambda x: POS_MAP[int(x)])(pos_arr).astype(np.int8)

    # Save NPZ with SS, Protein Class and PDB ID
    np.savez_compressed(
        out_npz,
        aa_ids=aa_arr,
        cluster_idx=cl_idx,
        sasa=sasa_arr,
        ss=ss_arr,                           # Secondary Structure field
        protein_class_id=protein_class_arr,  # Protein Class ID field
        pdb_id=pdb_id_arr,                   # PDB ID field
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
            "note": "IDs run from 0..19; amino acids in alphabetical order."
        }, f, indent=2)
    
    # Save SS mapping
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
    
    # Save Protein Class mapping
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
    print(f"Saved: {out_npz}")
    print(f"{'='*60}")
    print(f"Windows N = {aa_arr.shape[0]}, clusters K = {len(unique_clusters)}")
    
    print(f"\nSASA statistics:")
    print(f"  Min:    {sasa_arr.min():.4f}")
    print(f"  Max:    {sasa_arr.max():.4f}")
    print(f"  Mean:   {sasa_arr.mean():.4f}")
    print(f"  Median: {np.median(sasa_arr):.4f}")
    
    # SS statistics
    print(f"\nSECONDARY STRUCTURE statistics:")
    print(f"  Type: auto-detected")
    print(f"  Classes: {n_ss_classes}")
    print(f"  Missing SS values: {ss_missing_count}/{total_residues} ({ss_missing_count/total_residues*100:.1f}%)")
    
    unique_ss, counts_ss = np.unique(ss_arr, return_counts=True)
    print(f"  Distribution:")
    for ss_idx, count in zip(unique_ss, counts_ss):
        ss_char = idx_to_ss.get(ss_idx, '?')
        percentage = count / ss_arr.size * 100
        print(f"    Class {ss_idx} ('{ss_char}'): {count:6d} ({percentage:5.1f}%)")
    
    # Protein Class statistics
    print(f"\nPROTEIN CLASS statistics:")
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
    
    # PDB ID statistics
    print(f"\nPDB ID statistics:")
    unique_pdbs = set(pdb_id_arr)
    print(f"  Unique PDB IDs: {len(unique_pdbs)}")
    print(f"  Missing PDB IDs: {pdb_missing_count}/{len(all_windows)} ({pdb_missing_count/len(all_windows)*100:.1f}%)")
    
    print(f"\nMappings saved in: {out_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
