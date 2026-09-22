#!/usr/bin/env python3
"""
Build 5-residue sliding windows along beta strands, assigning each residue
a rotamer cluster id (based on its chi1/chi2 side-chain dihedral angles)
and carrying over its secondary structure (SS) and solvent accessibility
(ACC_norm/Exposure) from an input CSV.

For each PDB structure, residues are matched against a CSV of
per-residue classifications (chain, residue number, secondary structure,
exposure/ACC_norm), grouped into contiguous beta strands, and then
scanned with a window of size WINDOW (default 5) and step STEP (default
1) to produce overlapping 5-residue windows centered on each eligible
residue.

Usage:
    python script.py residui_classificati.csv <pdb_dir>

    residui_classificati.csv : CSV with per-residue classification data
        (must include at least PDB, Chain, ResNum, and typically SS,
        Exposure, ACC_norm columns).
    pdb_dir                  : Directory containing the corresponding
        .pdb structure files.

Outputs:
    <pdb_stem>_windows.csv : One CSV per processed PDB structure with the
        flattened 5-residue windows for that structure.
    windows_all.csv         : All windows from every processed structure,
        concatenated into a single CSV.
    windows_all.json        : The same windows in JSON form (full,
        non-flattened residue dicts, including per-residue "win_pos").
"""
from __future__ import annotations
import json, math, sys, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from Bio.PDB import PDBParser, Polypeptide
from Bio.PDB.vectors import calc_dihedral

WINDOW = 5
STEP   = 1  
# rotamer_localmax_maxdens.csv lives in ../common (a dependency shared with model/)
ROT_LIB_PATH = Path(__file__).resolve().parent.parent / "common" / "rotamer_localmax_maxdens.csv"

# 1. Atom table used to compute chi angles
CHI_ATOMS: Dict[str, Dict[str, List[str]]] = {
    "SER": {"chi1": ["N", "CA", "CB", "OG"]},
    "THR": {"chi1": ["N", "CA", "CB", "OG1"]},
    "CYS": {"chi1": ["N", "CA", "CB", "SG"]},
    "VAL": {"chi1": ["N", "CA", "CB", "CG1"]},
    "ILE": {"chi1": ["N", "CA", "CB", "CG1"], "chi2": ["CA", "CB", "CG1", "CD1"]},
    "LEU": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "CD1"]},
    "ASP": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "OD1"]},
    "ASN": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "OD1"]},
    "GLU": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "CD"]},
    "GLN": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "CD"]},
    "PHE": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "CD1"]},
    "TYR": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "CD1"]},
    "TRP": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "CD1"]},
    "HIS": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "ND1"]},
    "LYS": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "CD"],
            "chi3": ["CB", "CG", "CD", "CE"], "chi4": ["CG", "CD", "CE", "NZ"]},
    "ARG": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "CD"],
            "chi3": ["CB", "CG", "CD", "NE"], "chi4": ["CG", "CD", "NE", "CZ"]},
    "MET": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "SD"],
            "chi3": ["CB", "CG", "SD", "CE"]},
    "PRO": {"chi1": ["N", "CA", "CB", "CG"],  "chi2": ["CA", "CB", "CG", "CD"]},
}

LibTuple = Tuple[int, float, Optional[float], float]
RotLib   = Dict[str, List[LibTuple]]

def load_rotamer_library(csv_path: Path) -> RotLib:
    """
    Loads the rotamer library CSV into a dict keyed by 3-letter residue
    name, mapping each residue to its list of known rotamer clusters,
    sorted by decreasing peak density.

    Args:
        csv_path: Path to the rotamer library CSV. Expected columns:
            resname, cluster_id, chi1, chi2 (optional/NaN allowed),
            density_max.

    Returns:
        RotLib: dict mapping residue name (e.g. "PHE") to a list of
        (cluster_id, chi1, chi2_or_None, density_max) tuples. Returns an
        empty dict if csv_path does not exist.
    """
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    lib: RotLib = {}
    for _, row in df.iterrows():
        res = row.resname.strip().upper()
        lib.setdefault(res, []).append((
            int(row.cluster_id),
            float(row.chi1),
            None if pd.isna(row.get("chi2")) else float(row.chi2),
            float(row.density_max),
        ))
    for res in lib:
        lib[res].sort(key=lambda t: t[3], reverse=True)
    return lib

def ang_dist(a, b):
    """Returns the shortest angular distance (in degrees) between two angles."""
    return min(abs(a - b), 360 - abs(a - b))

def assign_cluster(res, chi1, chi2, lib: RotLib):
    """
    Assigns a rotamer cluster id to a residue based on its chi1/chi2
    angles, by finding the closest cluster centroid in the rotamer
    library.

    Args:
        res: 3-letter residue name (e.g. "PHE").
        chi1: Measured chi1 dihedral angle in degrees, or None if
            unavailable.
        chi2: Measured chi2 dihedral angle in degrees, or None if
            unavailable/not applicable to this residue.
        lib: Rotamer library, as returned by `load_rotamer_library`.

    Returns:
        int: 1-based cluster id (0 if no cluster could be assigned, e.g.
        residue not in the library or no angle available; ALA/GLY always
        return 1, since they have no side-chain rotamers).
    """
    res = res.strip().upper()
    if res in ["ALA", "GLY"]: return 1
    if res not in lib: return 0
    best, best_s = None, 1e9
    for cid, c1, c2, _ in lib[res]:
        s, n = 0, 0
        if chi1 is not None and c1 is not None:
            s += ang_dist(chi1, c1); n += 1
        if chi2 is not None and c2 is not None:
            s += ang_dist(chi2, c2); n += 1
        if n and s < best_s:
            best, best_s = cid, s
    return best+1 if best is not None else 0

def dih(a, b, c, d):
    """Computes the dihedral angle (in degrees) defined by four Biopython atoms."""
    return math.degrees(calc_dihedral(a.get_vector(), b.get_vector(),
                                      c.get_vector(), d.get_vector()))

def get_chis(res):
    """
    Computes the chi1 and chi2 side-chain dihedral angles for a residue,
    using the atom definitions in CHI_ATOMS.

    Args:
        res: A Biopython Residue object.

    Returns:
        tuple (chi1, chi2): Each is a float in degrees, or None if the
        residue type has no such angle defined, the required atoms are
        missing, or the dihedral could not be computed.
    """
    resname = res.get_resname().strip().upper()
    spec = CHI_ATOMS.get(resname, {})
    chis: List[Optional[float]] = []
    for chi_key in ("chi1", "chi2", "chi3", "chi4"):
        atoms = spec.get(chi_key)
        if atoms and all(a in res for a in atoms):
            try: angle = dih(*(res[a] for a in atoms))
            except: angle = None
        else: angle = None
        chis.append(angle)
    return chis[0], chis[1]

def split_into_strands(all_res):
    """
    Splits a flat, chain/residue-number-ordered list of residue dicts
    into separate contiguous strands (runs of consecutive residue
    numbers within the same chain).

    Args:
        all_res: List of residue dicts, each with "Chain" and "ResNum"
            keys, sorted by Chain then ResNum.

    Returns:
        List of lists, each inner list a contiguous run of residues
        belonging to the same chain with consecutive residue numbers.
    """
    if not all_res: return []
    strands = [[all_res[0]]]
    for prev, cur in zip(all_res, all_res[1:]):
        if cur["Chain"] == prev["Chain"] and cur["ResNum"] == prev["ResNum"] + 1:
            strands[-1].append(cur)
        else: strands.append([cur])
    return strands

# --------------------------------------------------------------------------------
# UPDATED: added ss{prefix} to the flattened row
# --------------------------------------------------------------------------------
def flatten_window(win: dict, window_id: int, size: int = WINDOW) -> dict:
    """
    Flattens a single window dict (with a nested "residues" list) into a
    single-row dict suitable for a CSV/DataFrame, with one set of
    per-residue columns (chain, resnum, AA, ss, chi1, chi2, conf, sasa)
    for each position in the window, numbered 1..size.

    Args:
        win: Window dict with keys "pdb", "strand_start", "face",
            "length", and "residues" (a list of per-residue dicts).
        window_id: Integer id to store under the "window_id" column.
        size: Expected window size (default: WINDOW). Positions beyond
            the number of residues actually present are filled with
            None.

    Returns:
        dict: A flat row ready to be turned into a DataFrame row.
    """
    row = {
        "window_id": window_id,
        "pdb": win["pdb"],
        "strand_start": win["strand_start"],
        "face": win["face"],
        "length": win["length"],
    }
    lst = win["residues"]
    for i in range(size):
        prefix = f"{i+1}"
        if i < len(lst):
            r = lst[i]
            row[f"chain{prefix}"]  = r["Chain"]
            row[f"resnum{prefix}"] = r["ResNum"]
            row[f"AA{prefix}"]     = r["AA"]
            row[f"ss{prefix}"]     = r.get("SS")       
            row[f"chi1_{prefix}"]  = r["chi1"]
            row[f"chi2_{prefix}"]  = r["chi2"]
            row[f"conf{prefix}"]   = r["cluster_id"]
            row[f"sasa{prefix}"]   = r.get("ACC_norm")
        else:
            row[f"chain{prefix}"]  = None
            row[f"resnum{prefix}"] = None
            row[f"AA{prefix}"]     = None
            row[f"ss{prefix}"]     = None              
            row[f"chi1_{prefix}"]  = None
            row[f"chi2_{prefix}"]  = None
            row[f"conf{prefix}"]   = None
            row[f"sasa{prefix}"]   = None
    return row

# --------------------------------------------------------------------------------
# UPDATED: store SS in the per-residue dict
# --------------------------------------------------------------------------------
def process_pdb(pdb_id, df_pdb, pdb_path: Path, lib: RotLib, start_id: int = 0):
    """
    Processes a single PDB structure: matches its residues against the
    per-residue classification rows in `df_pdb`, computes chi angles and
    rotamer cluster ids, splits residues into contiguous beta strands,
    and slides a 5-residue window along each strand to build the output
    windows.

    Args:
        pdb_id: Identifier to use for this structure (e.g. the file
            stem), stored as the "pdb" field of each window.
        df_pdb: DataFrame slice with the classification rows for this
            PDB only (must include Chain, ResNum, and typically SS,
            Exposure, ACC_norm columns).
        pdb_path: Path to the .pdb structure file to parse.
        lib: Rotamer library, as returned by `load_rotamer_library`.
        start_id: Starting value for the global, cross-structure window
            id counter (default: 0).

    Returns:
        tuple:
            df_out (pd.DataFrame): One flattened row per window produced
                for this structure (empty DataFrame on parse failure).
            windows_out (list[dict]): The same windows, in full
                (non-flattened) form, each including a "residues" list.
            current_id (int): The next available global window id, to be
                passed as `start_id` for the next structure.
    """
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try: structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as e:
        print(f"[SKIP] {pdb_id}: parsing - {e}", file=sys.stderr)
        return pd.DataFrame(), [], start_id

    model = structure[0]
    res_map = {(r.get_parent().id, r.id[1]): r for r in model.get_residues()
               if Polypeptide.is_aa(r, standard=True)}

    beta = []
    for _, row in df_pdb.sort_values(["Chain", "ResNum"]).iterrows():
        key = (row.Chain.strip(), int(row.ResNum))
        res = res_map.get(key)
        if res is None: continue
        
        res3 = res.get_resname().strip().upper() 
        chi1, chi2 = get_chis(res)
        cid = assign_cluster(res3, chi1, chi2, lib)
        
        beta.append({
            "Chain": row.Chain.strip(),
            "ResNum": int(row.ResNum),
            "AA": res3,
            "SS": row.get("SS"),              
            "Exposure": row.get("Exposure"),
            "ACC_norm": row.get("ACC_norm"),
            "chi1": None if chi1 is None else round(chi1, 1),
            "chi2": None if chi2 is None else round(chi2, 1),
            "cluster_id": cid,
        })

    windows_out = []
    OFFS = (-2, -1, 0, 1, 2)
    strands = split_into_strands(beta)
    
    current_id = start_id
    for strand in strands:
        if len(strand) < WINDOW: continue
        for i in range(2, len(strand) - 2):
            w = [strand[i-2], strand[i-1], strand[i], strand[i+1], strand[i+2]]
            face_central = w[2]["Exposure"]
            residues = [{**r, "win_pos": off} for r, off in zip(w, OFFS)]
            windows_out.append({
                "window_id": current_id,
                "pdb": pdb_id,
                "strand_start": strand[0]["ResNum"],
                "face": face_central,
                "length": 5,
                "residues": residues,
            })
            current_id += 1

    rows = [flatten_window(w, w["window_id"]) for w in windows_out]
    return pd.DataFrame(rows), windows_out, current_id

def resolve_pdb_path(root: Path, pdb_id: str):
    """
    Resolves a PDB identifier (as it appears in the classification CSV)
    to an actual .pdb file under `root`, trying a series of increasingly
    permissive matching strategies (exact name, "_new" suffix toggled,
    4-character prefix match).

    Args:
        root: Directory to search for .pdb files.
        pdb_id: PDB identifier/filename stem from the CSV (with or
            without a .pdb extension).

    Returns:
        tuple (path, how): `path` is the resolved Path, or None if no
        match was found; `how` is a short string describing which
        strategy succeeded (or "not_found").
    """
    base = pdb_id[:-4] if pdb_id.lower().endswith(".pdb") else pdb_id
    p = root / (base + ".pdb")
    if p.exists(): return p, "exact"
    if base.endswith("_new"):
        cand = root / (base[:-4] + ".pdb")
        if cand.exists(): return cand, "toggle_new_removed"
    else:
        cand = root / (base + "_new.pdb")
        if cand.exists(): return cand, "toggle_new_added"
    m = re.match(r"([0-9A-Za-z]{4})", base)
    if m:
        prefix = m.group(1)
        candidates = sorted(root.glob(f"{prefix}*.pdb"))
        if len(candidates) == 1: return candidates[0], "prefix_match"
        if len(candidates) > 1:
            new_ones = [c for c in candidates if c.name.endswith("_new.pdb")]
            chosen = new_ones[0] if new_ones else max(candidates, key=lambda x: x.stat().st_mtime)
            return chosen, f"prefix_match_many({len(candidates)})"
    return None, "not_found"

def main(csv_file, pdb_dir):
    """
    Entry point: reads the classification CSV, groups it by PDB id,
    resolves and processes each corresponding structure file, and writes
    both per-structure and combined (CSV + JSON) window outputs.

    Args:
        csv_file: Path to the per-residue classification CSV (must
            include a "PDB" column to group rows by structure).
        pdb_dir: Directory containing the .pdb files referenced by the
            "PDB" column.
    """
    df = pd.read_csv(csv_file)
    lib = load_rotamer_library(ROT_LIB_PATH)
    root = Path(pdb_dir)
    all_dfs, all_json = [], []
    global_window_id = 1

    for pdb_id, grp in df.groupby("PDB"):
        pdb_path, how = resolve_pdb_path(root, pdb_id)
        if not pdb_path:
            print(f"[SKIP] {pdb_id}: PDB file not found")
            continue

        df_out, json_out, global_window_id = process_pdb(
            pdb_path.stem, grp, pdb_path, lib, global_window_id
        )
        if df_out is None or df_out.empty: continue

        out_csv = pdb_path.with_name(pdb_path.stem + "_windows.csv")
        df_out.to_csv(out_csv, index=False)
        all_dfs.append(df_out)
        all_json.extend(json_out)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_csv("windows_all.csv", index=False)
        with open("windows_all.json", "w", encoding="utf-8") as f:
            json.dump(all_json, f, indent=2)
        print(f"Created windows_all.csv with {len(df_all)} windows.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python script.py residui_classificati.csv <pdb_dir>")
    main(sys.argv[1], sys.argv[2])
