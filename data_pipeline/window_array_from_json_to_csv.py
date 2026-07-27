#!/usr/bin/env python3
from __future__ import annotations
import json, math, sys, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from Bio.PDB import PDBParser, Polypeptide
from Bio.PDB.vectors import calc_dihedral

WINDOW = 5
STEP   = 1  
# rotamer_localmax_maxdens.csv vive in ../common (dipendenza condivisa con model/)
ROT_LIB_PATH = Path(__file__).resolve().parent.parent / "common" / "rotamer_localmax_maxdens.csv"

# 1. Tabella atomi per χ-angles
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
    return min(abs(a - b), 360 - abs(a - b))

def assign_cluster(res, chi1, chi2, lib: RotLib):
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
    return math.degrees(calc_dihedral(a.get_vector(), b.get_vector(),
                                      c.get_vector(), d.get_vector()))

def get_chis(res):
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
    if not all_res: return []
    strands = [[all_res[0]]]
    for prev, cur in zip(all_res, all_res[1:]):
        if cur["Chain"] == prev["Chain"] and cur["ResNum"] == prev["ResNum"] + 1:
            strands[-1].append(cur)
        else: strands.append([cur])
    return strands

# --------------------------------------------------------------------------------
# MODIFICATO: aggiunto ss{prefix} nel flattening
# --------------------------------------------------------------------------------
def flatten_window(win: dict, window_id: int, size: int = WINDOW) -> dict:
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
            row[f"ss{prefix}"]     = r.get("SS")        # <--- AGGIUNTO
            row[f"chi1_{prefix}"]  = r["chi1"]
            row[f"chi2_{prefix}"]  = r["chi2"]
            row[f"conf{prefix}"]   = r["cluster_id"]
            row[f"sasa{prefix}"]   = r.get("ACC_norm")
        else:
            row[f"chain{prefix}"]  = None
            row[f"resnum{prefix}"] = None
            row[f"AA{prefix}"]     = None
            row[f"ss{prefix}"]     = None               # <--- AGGIUNTO
            row[f"chi1_{prefix}"]  = None
            row[f"chi2_{prefix}"]  = None
            row[f"conf{prefix}"]   = None
            row[f"sasa{prefix}"]   = None
    return row

# --------------------------------------------------------------------------------
# MODIFICATO: salvataggio SS nel dizionario del residuo
# --------------------------------------------------------------------------------
def process_pdb(pdb_id, df_pdb, pdb_path: Path, lib: RotLib, start_id: int = 0):
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try: structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as e:
        print(f"[SKIP] {pdb_id}: parsing – {e}", file=sys.stderr)
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
            "SS": row.get("SS"),              # <--- AGGIUNTO legge dal CSV
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
    df = pd.read_csv(csv_file)
    lib = load_rotamer_library(ROT_LIB_PATH)
    root = Path(pdb_dir)
    all_dfs, all_json = [], []
    global_window_id = 1

    for pdb_id, grp in df.groupby("PDB"):
        pdb_path, how = resolve_pdb_path(root, pdb_id)
        if not pdb_path:
            print(f"[SKIP] {pdb_id}: PDB non trovato")
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
        print(f"Creato windows_all.csv con {len(df_all)} finestre.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Uso: python script.py residui_classificati.csv <pdb_dir>")
    main(sys.argv[1], sys.argv[2])