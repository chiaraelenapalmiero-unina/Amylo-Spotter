"""
Mutation probability heatmap + full mutational analysis
=========================================================

Extends mutation_heatmap1_or.py with:

  1. Probability heatmap (original, unchanged)
  2. DeltaLogP mutational landscape  (20xN matrix, per window)
  3. Entropy profile              (rigidity vs flexibility per position)
  4. Breaker mutations            (DeltaLogP < threshold, default -5)
  5. Designable positions         (tolerated mutations, |DeltaLogP| < 1)
  6. Hotspot detection            (entropy > 75th percentile)
  7. Fibril core score            (mean |DeltaLogP|, conserved positions)
  8. UMAP/PCA embedding           (structural clustering of windows)
  9. Summary report               (JSON + CSV)

Usage:
    python mutation_heatmap_extended.py \\
        --model    teachers/teacher_alpha-synuclein/best_teacher.pt \\
        --data     windows_all_local.npz \\
        --protein_metadata proteins_metadata_with_class.json \\
        --rotamer_csv rotamer_localmax_maxdens.csv \\
        --fasta    fasta1.fasta \\
        --window_seq VGGAV \\
        --output   results/asynuclein_VGGAV
        python mutation_heatmap_extended_old.py        --model teachers/teacher_/best_teacher.pt        --protein_name "alpha-synuclein"       --data     windows_all_local.npz         --protein_metadata proteins_metadata_with_class.json         --rotamer_csv rotamer_localmax_maxdens.csv         --fasta  ./fasta_file/aS_7v49.fasta         --window_seq VGGAV         --output   resultsalpha_syn_VVGAV --sequence_scan --start_residue 38  --end_residue 95
        
        python mutation_heatmap_extended_old.py \
    --model    teachers/teacher_Transthyretin/best_teacher.pt \
    --data     windows_all_local.npz \
    --protein_metadata proteins_metadata_with_class.json \
    --rotamer_csv rotamer_localmax_maxdens.csv \
    --fasta    ./fasta_file/Transthyretin_8ADE.fasta \
    --protein_name "Transthyretin" \
    --window_seq AALLS \
    --output   results/ttr \
    --sequence_scan


    All outputs are saved to the --output folder:
        heatmap_prob.png          <- probability heatmap (original)
        heatmap_delta_logp.png    <- DeltaLogP landscape
        entropy_profile.png       <- entropy per window
        core_score.png            <- fibril core score
        breaker_map.png           <- breaker mutations map
        embedding_umap.png        <- structural clustering (if sklearn/umap available)
        analysis_report.json      <- full numeric summary
        analysis_report.csv       <- per-position table

        set -euo pipefail
 
MODEL=teacher_pathological_vs_functional/best_teacher.pt
DATA=windows_all_local.npz
META=proteins_metadata_with_class.json
ROTAMER=rotamer_localmax_maxdens.csv
 
# format: protein_name|fasta|initial_window_seq
declare -a PROTEINS=(
  "Antimicrobial peptide|./fasta_file/Antimicrobial peptide_7QV5.fasta|DLIRK"
  "CPEB3|./fasta_file/CPEB3_8SPA.fasta|LSPSF"
  "CsgA|./fasta_file/CsgA_8ENQ.fasta|PNSEL"
  "HET-s (218-289)|./fasta_file/HET-s_2RNM.fasta|MKIDA"
  "Mrip3|./fasta_file/Mrip3_6JPD.fasta|VFNNC"
  "Nup98|./fasta_file/Nup98_7Q64.fasta|TGTAN"
  "RIPK1|./fasta_file/RIPK1_8IB0.fasta|TIYNS"
  "beta-endorphin|./fasta_file/beta-endorphin_6TUB.fasta|YGGFM"
)
 
for entry in "${PROTEINS[@]}"; do
  IFS='|' read -r NAME FASTA WINSEQ <<< "$entry"
  SAFE_NAME=$(echo "$NAME" | tr ' /()' '____')
 
  echo "=== $NAME ==="
  python mutation_heatmap_extended_old.py \
    --model "$MODEL" \
    --data "$DATA" \
    --protein_metadata "$META" \
    --rotamer_csv "$ROTAMER" \
    --fasta "$FASTA" \
    --protein_name "$NAME" \
    --window_seq "$WINSEQ" \
    --output "results/functional_${SAFE_NAME}" \
    --sequence_scan
done

set -euo pipefail

MODEL=teacher_pathological_vs_functional/best_teacher.pt
DATA=windows_all_local.npz
META=proteins_metadata_with_class.json
ROTAMER=rotamer_localmax_maxdens.csv
SCRIPT=mutation_heatmap_extended_correction.py

# ---- HET-s ----
NAME="HET-s (218-289)"
FASTA="./fasta_file/HET-s_2RNM.fasta"
OFFSET=216
SAFE_NAME=$(echo "$NAME" | tr ' /()' '____')
declare -a WINS=(
  "T233|IRTEE"
  "G242|QLGNV"
  "V245|NVVTA"
  "A247|VTAAA"
  "G252|LHGGI"
  "G283|YGGKG"
)

for entry in "${WINS[@]}"; do
  IFS='|' read -r LABEL WIN <<< "$entry"
  echo "=== $NAME - $LABEL ($WIN) ==="
  python "$SCRIPT" \
    --model "$MODEL" \
    --data "$DATA" \
    --protein_metadata "$META" \
    --rotamer_csv "$ROTAMER" \
    --fasta "$FASTA" \
    --protein_name "$NAME - $LABEL" \
    --window_seq "$WIN" \
    --output "results/hotspots_${SAFE_NAME}/${LABEL}_${WIN}" \
    --residue_offset "$OFFSET"
done

# ---- Mrip3 ----
NAME="Mrip3"
FASTA="./fasta_file/Mrip3_6JPD.fasta"
OFFSET=402
SAFE_NAME=$(echo "$NAME" | tr ' /()' '____')
declare -a WINS=(
  "V448|SEVQI"
  "G451|QIGNY"
  "
)

for entry in "${WINS[@]}"; do
  IFS='|' read -r LABEL WIN <<< "$entry"
  echo "=== $NAME - $LABEL ($WIN) ==="
  python "$SCRIPT" \
    --model "$MODEL" \
    --data "$DATA" \
    --protein_metadata "$META" \
    --rotamer_csv "$ROTAMER" \
    --fasta "$FASTA" \
    --protein_name "$NAME - $LABEL" \
    --window_seq "$WIN" \
    --output "results/hotspots_${SAFE_NAME}/${LABEL}_${WIN}" \
    --residue_offset "$OFFSET"
done
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# topk_model2.py lives in ../model (used below in load_model())
sys.path.append(str(Path(__file__).resolve().parent.parent / "model"))


# -- Helper: multi-format saving -------------------------------------------
def _save_multiformat(fig, output_path: str, label: str = ""):
    """
    Saves the figure as PNG (600 dpi), SVG, PDF and TIFF (600 dpi, LZW)
    inside a 'figures/' subfolder next to output_path.
    """
    src  = Path(output_path)
    dest = src.parent / 'figures' / src.stem
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(dest) + '.png',  format='png',  dpi=600, bbox_inches='tight')
    fig.savefig(str(dest) + '.svg',  format='svg',  bbox_inches='tight')
    fig.savefig(str(dest) + '.pdf',  format='pdf',  bbox_inches='tight')
    fig.savefig(str(dest) + '.tiff', format='tiff', dpi=600,
                bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    tag = f"  [{label}]" if label else ""
    print(f"[OK]{tag} Saved: {dest}  (.png / .svg / .pdf / .tiff)")

# -- AA constants -------------------------------------------------------------
AA_ORDER   = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX  = {aa: i for i, aa in enumerate(AA_ORDER)}
IDX_TO_AA  = {i: aa for i, aa in enumerate(AA_ORDER)}
AA_NAMES   = {i: aa for i, aa in enumerate(AA_ORDER)}
AA_TO_NUM  = {v: k for k, v in AA_NAMES.items()}

# Biophysical properties for annotations (same order as AA_ORDER)
AA_PROPERTIES = {
    # Hydrophobicity (Kyte-Doolittle, normalized 0-1)
    'hydrophobic': ['A','C','F','I','L','M','V','W'],
    # Polar/charged
    'charged_pos': ['K','R','H'],
    'charged_neg': ['D','E'],
    'polar':       ['N','Q','S','T','Y'],
    # Special
    'proline':     ['P'],
    'glycine':     ['G'],
}

# Colors per property (used in annotations)
AA_PROP_COLOR = {}
for aa in AA_ORDER:
    if aa in AA_PROPERTIES['hydrophobic']:  AA_PROP_COLOR[aa] = '#1A3A64'
    elif aa in AA_PROPERTIES['charged_pos']: AA_PROP_COLOR[aa] = '#8B0000'
    elif aa in AA_PROPERTIES['charged_neg']: AA_PROP_COLOR[aa] = '#006400'
    elif aa in AA_PROPERTIES['polar']:       AA_PROP_COLOR[aa] = '#4B0082'
    elif aa in AA_PROPERTIES['proline']:     AA_PROP_COLOR[aa] = '#8B4513'
    elif aa in AA_PROPERTIES['glycine']:     AA_PROP_COLOR[aa] = '#2F4F4F'
    else:                                    AA_PROP_COLOR[aa] = '#333333'


# ============================================================================
# BLOCK 1 - Original functions (100% unchanged)
# ============================================================================

def read_fasta(fasta_path):
    """Reads a FASTA file and returns its single concatenated sequence string."""
    seq = ''
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    return seq


def _extract_pdb_id_from_fasta(fasta_path):
    """Extracts the PDB id from an RCSB-style header like '>2RNM_1|Chains...' -> '2RNM'."""
    try:
        with open(fasta_path) as f:
            for line in f:
                if line.startswith('>'):
                    token = line[1:].strip().split('|')[0]   # e.g. '2RNM_1'
                    pdb = token.split('_')[0].strip()
                    return pdb.upper() if pdb else None
    except Exception:
        pass
    return None


def _parse_sifts_offset(data, pdb_key):
    """
    Computes offset = author_number - SEQRES_position(=FASTA) from the SIFTS
    PDBe API response. Uses the longest UniProt segment. Returns int or None.
    """
    unp_block = data.get(pdb_key, {}).get('UniProt', {})
    best_len, offset = -1, None
    for _acc, acc_data in unp_block.items():
        for seg in acc_data.get('mappings', []):
            start    = seg.get('start', {})
            res_num  = start.get('residue_number')          # 1-based in SEQRES = FASTA pos
            auth_num = start.get('author_residue_number')   # deposited (canonical) numbering
            if res_num is None or auth_num is None:
                continue
            seg_len = seg.get('unp_end', 0) - seg.get('unp_start', 0)
            if seg_len > best_len:
                best_len = seg_len
                offset   = int(auth_num) - int(res_num)
    return offset


def auto_residue_offset(pdb_id, timeout=20, _cache={}):
    """
    Automatically determines the offset (canonical_residue - 1-based_FASTA_position)
    from the PDBe SIFTS mapping, using the AUTHOR residue number, i.e. the
    numbering deposited in the PDB (HET-s 218-289, RIPK3 409-486, beta-endorphin 1-31).
    Requires a connection to ebi.ac.uk. Returns int, or None if it cannot be determined.
    """
    if not pdb_id:
        return None
    key = pdb_id.lower()[:4]
    if key in _cache:
        return _cache[key]
    import json as _json
    import urllib.request as _url
    offset = None
    try:
        req = _url.Request(f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{key}",
                           headers={'User-Agent': 'mutation-heatmap/1.0'})
        with _url.urlopen(req, timeout=timeout) as resp:
            data = _json.load(resp)
        offset = _parse_sifts_offset(data, key)
    except Exception as e:
        print(f"  [WARN] auto-offset: SIFTS not reachable for {key} ({e}).")
        offset = None
    _cache[key] = offset
    return offset


def load_model(model_path, device):
    """
    Loads a ProteinMLM_v2 checkpoint and rebuilds the model with the
    architecture stored in its 'model_config' metadata.

    Args:
        model_path: Path to the .pt checkpoint file (must contain a
            'model_config' dict and a 'model_state_dict').
        device: Torch device to load the model onto.

    Returns:
        ProteinMLM_v2: The loaded model, in eval() mode.

    Raises:
        KeyError: If the checkpoint has no 'model_config' entry.
    """
    from topk_model2 import ProteinMLM_v2

    print(f"\nLoading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_config' in ckpt:
        cfg       = ckpt['model_config']
        state_key = 'model_state_dict'
    else:
        raise KeyError("The checkpoint does not contain 'model_config'.")

    n_aa            = cfg.get('n_aa', 20)
    aa_to_n_clusters = cfg.get('aa_to_n_clusters', {i: 1 for i in range(20)})
    n_pos           = cfg.get('n_pos', 3)
    n_ss            = cfg.get('n_ss', 8)
    d_embed         = cfg.get('d_embed', 256)
    d_hidden        = cfg.get('d_hidden', 512)
    n_lstm_layers   = cfg.get('n_lstm_layers', 3)
    dropout         = cfg.get('dropout', 0.25)
    embed_dropout   = cfg.get('embed_dropout', 0.3)
    use_blosum      = cfg.get('use_blosum', True)

    print(f"  n_aa={n_aa}, d_embed={d_embed}, d_hidden={d_hidden}, "
          f"lstm_layers={n_lstm_layers}, n_pos={n_pos}, n_ss={n_ss}")

    model = ProteinMLM_v2(
        n_aa=n_aa, aa_to_n_clusters=aa_to_n_clusters,
        n_pos=n_pos, n_ss=n_ss, d_embed=d_embed, d_hidden=d_hidden,
        n_lstm_layers=n_lstm_layers, dropout=dropout,
        embed_dropout=embed_dropout, use_blosum=use_blosum,
    ).to(device)

    model.load_state_dict(ckpt[state_key])
    model.eval()
    print("[OK] Model loaded")
    return model


def _normalize_pdb_id(pdb_id: str) -> str:
    """
    Reduces a pdb_id to the canonical 4-character PDB code (case-insensitive).
    Examples:
        '8BFA_filtered_new_r5c23d4.836a0.648'     -> '8bfa'
        '8bfa_filtered_new_r5c23d4.836a0.648_new' -> '8bfa'
        '2MPZ_filtered_new_r5c23d4.8a0.0'         -> '2mpz'
    """
    return pdb_id.split('_')[0].lower()


def load_dataset(npz_path, protein_metadata_path, target_protein):
    """
    Loads the windows dataset and filters it down to the windows
    belonging to `target_protein`, deduplicating any PDB structure that
    appears twice under a different naming convention.

    Args:
        npz_path: Path to the windows .npz dataset.
        protein_metadata_path: Path to the JSON metadata file mapping
            each window index to its pdb_id/protein_class.
        target_protein: The protein_class name to filter windows by.

    Returns:
        tuple (data, dedup_indices):
            data: The loaded .npz archive (NpzFile).
            dedup_indices: Sorted list of window indices belonging to
                `target_protein`, after removing duplicate-PDB windows.
    """
    print(f"\nLoading dataset: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    print(f"  NPZ keys: {list(data.keys())}")

    with open(protein_metadata_path) as f:
        protein_metadata = json.load(f)

    def get_protein_name(v):
        if isinstance(v, dict):
            return v.get('protein_class', '')
        return str(v)

    def get_pdb_id(v):
        if isinstance(v, dict):
            return v.get('pdb_id', '')
        return ''

    all_indices = [
        int(idx) for idx, val in protein_metadata.items()
        if get_protein_name(val) == target_protein
    ]
    print(f"  Total windows in dataset: {data['aa_ids'].shape[0]}")
    print(f"  Windows for '{target_protein}' (pre-dedup): {len(all_indices)}")

    # -- PDB deduplication ---------------------------------------------------
    # Some PDBs appear twice under different naming (e.g. '8BFA_filtered...'
    # and '8bfa_filtered..._new'). For each canonical PDB code (4 char,
    # lowercase) we keep ALL windows from the first version encountered and
    # discard ALL windows from the duplicate versions.
    #
    # Strategy:
    #   1. Group indices by full pdb_id  ->  {pdb_id: [idx, ...]}
    #   2. For each unique 4-char code, pick the version with the lowest
    #      index (reproducible, deterministic convention).
    #   3. Include only the indices of that version.

    # Step 1: group by full pdb_id
    pdb_id_to_indices: dict[str, list[int]] = {}
    for idx in all_indices:
        raw_pdb = get_pdb_id(protein_metadata.get(str(idx), {}))
        pdb_id_to_indices.setdefault(raw_pdb, []).append(idx)

    # Step 2: for each canonical code, choose the pdb_id with the lowest index
    code_to_chosen_pdb: dict[str, str] = {}   # 4-char code -> chosen pdb_id
    for raw_pdb in pdb_id_to_indices:
        code = _normalize_pdb_id(raw_pdb) if raw_pdb else f'__unknown__'
        min_idx = min(pdb_id_to_indices[raw_pdb])
        if code not in code_to_chosen_pdb:
            code_to_chosen_pdb[code] = raw_pdb
        else:
            current_min = min(pdb_id_to_indices[code_to_chosen_pdb[code]])
            if min_idx < current_min:
                code_to_chosen_pdb[code] = raw_pdb

    # Step 3: collect the indices of the chosen versions
    chosen_pdb_ids = set(code_to_chosen_pdb.values())
    dedup_indices  = []
    skipped_versions: list[str] = []

    for raw_pdb, idxs in sorted(pdb_id_to_indices.items(),
                                 key=lambda kv: min(kv[1])):
        if raw_pdb in chosen_pdb_ids:
            dedup_indices.extend(sorted(idxs))
        else:
            code = _normalize_pdb_id(raw_pdb)
            skipped_versions.append(
                (raw_pdb, len(idxs), code_to_chosen_pdb[code])
            )

    dedup_indices.sort()

    if skipped_versions:
        print(f"  [WARN] PDB deduplication: discarded {len(skipped_versions)} duplicate versions:")
        for dup_pdb, n_win, kept_pdb in skipped_versions:
            print(f"       REMOVED  ({n_win:3d} windows): '{dup_pdb}'")
            print(f"       KEPT                 : '{kept_pdb}'")
    else:
        print(f"  [OK] No duplicate PDB found.")

    print(f"  Windows for '{target_protein}' (post-dedup): {len(dedup_indices)}  "
          f"({len(code_to_chosen_pdb)} unique PDB structures)")

    return data, dedup_indices


def window_to_seq(aa_array):
    """Converts an array of amino acid indices for one window into its 1-letter sequence string."""
    return ''.join(AA_NAMES.get(int(a), '?') for a in aa_array)


def find_matching_windows(data, asynuclein_indices, window_seq):
    """
    Finds all windows among the given indices whose amino acid sequence
    exactly matches `window_seq`.

    Args:
        data: The loaded .npz dataset.
        asynuclein_indices: List of candidate window indices to search
            within (despite the name, this works for any protein).
        window_seq: The 5-letter amino acid sequence to match exactly.

    Returns:
        tuple (matched_indices, positions_info):
            matched_indices: List of window indices whose sequence equals
                `window_seq`.
            positions_info: List of dicts {'global_idx', 'sequence'} for
                each match.
    """
    matched_indices = []
    positions_info  = []
    aa_ids          = data['aa_ids']

    for idx in asynuclein_indices:
        seq = window_to_seq(aa_ids[idx])
        if seq == window_seq:
            matched_indices.append(idx)
            positions_info.append({'global_idx': idx, 'sequence': seq})

    print(f"\n  Windows matching '{window_seq}': {len(matched_indices)}")
    return matched_indices, positions_info


@torch.no_grad()
def predict_probs_single(model, data, global_idx, device, mask_token_aa, mask_token_cl):
    """
    Runs a forward pass for one window, masking the central residue's
    amino acid and cluster, and returns the predicted amino acid
    probability distribution for that position.

    Args:
        model: The loaded ProteinMLM_v2 model.
        data: The .npz dataset (must contain aa_ids, cluster_idx, ss,
            pos_idx, sasa).
        global_idx: Index of the window in the dataset arrays.
        device: Torch device to run inference on.
        mask_token_aa: Integer index used as the MASK token for the
            amino acid channel.
        mask_token_cl: Integer index used as the MASK token for the
            cluster channel.

    Returns:
        np.ndarray: Softmax probability distribution of shape (20,) over
        the 20 amino acids for the masked central residue.
    """
    win_size = data['aa_ids'].shape[1]
    center   = win_size // 2

    aa   = torch.tensor(data['aa_ids'][global_idx],     dtype=torch.long).unsqueeze(0).to(device)
    cl   = torch.tensor(data['cluster_idx'][global_idx], dtype=torch.long).unsqueeze(0).to(device)
    ss   = torch.tensor(data['ss'][global_idx],          dtype=torch.long).unsqueeze(0).to(device)
    pos  = torch.tensor(data['pos_idx'][global_idx],     dtype=torch.long).unsqueeze(0).to(device)
    sasa = torch.tensor(data['sasa'][global_idx],        dtype=torch.float).unsqueeze(0).to(device)

    aa[:, center] = mask_token_aa
    cl[:, center] = mask_token_cl

    batch = {
        'aa': aa, 'cl': cl, 'ss': ss, 'pos': pos, 'sasa': sasa,
        'y_aa': torch.zeros(1, dtype=torch.long).to(device),
        'y_cl': torch.zeros(1, dtype=torch.long).to(device),
    }

    output = model(batch)
    if isinstance(output, dict):    logits = output['logits_aa']
    elif isinstance(output, (tuple, list)): logits = output[0]
    else:                           logits = output

    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    return probs


def compute_probability_matrix(model, data, matched_indices, device,
                                mask_token_aa=20, mask_token_cl=0):
    """
    Computes the amino acid probability distribution for every matched
    window and stacks them into a single matrix.

    Args:
        model: The loaded ProteinMLM_v2 model.
        data: The .npz dataset.
        matched_indices: List of window indices to run inference on.
        device: Torch device to run inference on.
        mask_token_aa: MASK token index for the amino acid channel.
        mask_token_cl: MASK token index for the cluster channel.

    Returns:
        np.ndarray: Matrix of shape (20, len(matched_indices)), one
        probability column per window.
    """
    all_probs = []
    print(f"\nComputing probabilities for {len(matched_indices)} windows...")

    for i, idx in enumerate(matched_indices):
        probs = predict_probs_single(model, data, idx, device,
                                     mask_token_aa, mask_token_cl)
        all_probs.append(probs)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(matched_indices)}] idx={idx} [OK]")

    matrix = np.column_stack(all_probs)   # (20, N)
    print(f"  Final matrix: {matrix.shape[0]} AA x {matrix.shape[1]} windows")
    return matrix


def build_column_labels(matched_indices, data, fasta_seq, window_seq,
                        protein_metadata=None, residue_offset=0):
    """
    Builds a display label for each matched window: which residue is
    masked, its sequence position(s) in the FASTA, and the source PDB
    id, for use as x-axis tick labels in the plots.

    Args:
        matched_indices: List of window indices matching `window_seq`.
        data: The .npz dataset.
        fasta_seq: Full protein sequence string, or None if unavailable.
        window_seq: The 5-letter window sequence being labeled.
        protein_metadata: Optional dict mapping window index (as string)
            to its metadata (used to show the PDB id).
        residue_offset: Offset added to the FASTA-derived position to
            report it in canonical residue numbering.

    Returns:
        list[str]: One multi-line label per window, e.g.
        "Poly-1\\nV53\\n8BFA".
    """
    win_size = data['aa_ids'].shape[1]
    center   = win_size // 2
    aa_ids   = data['aa_ids']
    labels   = []

    fasta_positions = []
    if fasta_seq:
        start = 0
        while True:
            pos = fasta_seq.find(window_seq, start)
            if pos == -1:
                break
            fasta_positions.append(pos + center + 1 + residue_offset)
            start = pos + 1

    for i, idx in enumerate(matched_indices):
        center_aa_idx = int(aa_ids[idx, center])
        center_aa     = AA_NAMES.get(center_aa_idx, '?')

        if fasta_positions:
            fasta_pos = fasta_positions[i % len(fasta_positions)]
            pos_str   = f"{center_aa}{fasta_pos}"
        else:
            pos_str = center_aa

        pdb_str = ''
        if protein_metadata and str(idx) in protein_metadata:
            raw_pdb = protein_metadata[str(idx)]
            if isinstance(raw_pdb, dict):
                raw_pdb = raw_pdb.get('pdb_id', '')
            pdb_str = str(raw_pdb)[:4].upper()

        if pdb_str:
            labels.append(f"Poly-{i+1}\n{pos_str}\n{pdb_str}")
        else:
            labels.append(f"Poly-{i+1}\n{pos_str}")

    return labels


def plot_heatmap(matrix, column_labels, window_seq, output_path=None,
                 figsize=(18, 10), colormap='RdPu', protein_name='Protein'):
    """Original probability heatmap -- unchanged."""
    row_labels = [AA_NAMES[i] for i in range(20)]

    bounds = [0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.25, 0.30, 0.35, 0.45, 0.60]
    norm   = BoundaryNorm(bounds, ncolors=256)

    colors_list = [
        '#FFFFFF', '#FDF0F0', '#FAE1E1', '#F5C8C8', '#EFB0B0', '#E68F8F',
        '#DC6E6E', '#CC5050', '#B83A3A', '#A02D2D', '#8C2323', '#641A1A', '#501112'
    ]
    cmap_obj = LinearSegmentedColormap.from_list('asynuclein_reds', colors_list, N=256)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    im = ax.imshow(matrix, aspect='auto', cmap=cmap_obj,
                   interpolation='nearest', norm=norm)

    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5, minor=True)
    ax.set_yticks(np.arange(20) + 0.5, minor=True)
    ax.grid(which='minor', color='#D3D3D3', linestyle='-', linewidth=0.5)
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(20 - 0.5, -0.5)
    ax.tick_params(which='minor', size=0)

    for c in range(matrix.shape[1]):
        sorted_desc = np.argsort(matrix[:, c])[::-1]
        rank_map    = {int(aa_idx): rank + 1 for rank, aa_idx in enumerate(sorted_desc)}
        for r in range(matrix.shape[0]):
            val      = matrix[r, c]
            color    = 'white' if val >= 0.15 else 'black'
            rank_num = rank_map[int(r)]
            ax.text(c, r - 0.19, f'#{rank_num}',
                    ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')
            ax.text(c, r + 0.19, f'{val:.2f}',
                    ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha='right',
                   rotation_mode='anchor', fontsize=11)
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(row_labels, fontsize=14)
    ax.set_xlabel(f'{protein_name} Polymorphs', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Amino acid', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Mutation Probability - {protein_name}\nWindow: {window_seq}  '
        f'({matrix.shape[1]} occurrences in dataset)',
        fontsize=14, fontweight='bold', pad=20
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Predicted probability', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()

    if output_path:
        _save_multiformat(fig, output_path, label="heatmap_prob")

    plt.close(fig)


def print_summary(matrix, column_labels):
    """Prints the top-5 amino acids by average predicted probability and basic matrix stats."""
    mean_probs = matrix.mean(axis=1)
    top5_idx   = np.argsort(mean_probs)[::-1][:5]
    print("\n-- Summary of average probabilities (masked central residue) --")
    print(f"{'AA':<6} {'Mean prob':>12}")
    print("-" * 20)
    for idx in top5_idx:
        print(f"  {AA_NAMES[idx]:<4}  {mean_probs[idx]:.4f}")
    print(f"\nMatrix statistics:")
    print(f"  Min:  {matrix.min():.4f}  Max: {matrix.max():.4f}  "
          f"Mean: {matrix.mean():.4f}")


# ============================================================================
# BLOCK 2 - New analyses
# ============================================================================

def compute_delta_logp(prob_matrix, wt_sequence):
    """
    Computes the DeltaLogP matrix for every window.

    DeltaLogP(i, a) = log P(a | context_i) - log P(WT_i | context_i)

    Args:
        prob_matrix  : np.array (20, N_windows)
        wt_sequence  : window sequence string, e.g. "VGGAV"
                       The CENTRAL residue is used as the WT for every
                       window (all windows share the same central
                       sequence).
    
    Returns:
        delta_logp  : np.array (20, N_windows)
        wt_idx      : int -- index of the WT central amino acid
    """
    center     = len(wt_sequence) // 2
    wt_aa      = wt_sequence[center]
    wt_idx     = AA_TO_IDX[wt_aa]

    # log P(mut) - log P(WT) for each column (window)
    log_probs  = np.log(prob_matrix + 1e-9)          # (20, N)
    wt_log     = log_probs[wt_idx, :]                # (N,)
    delta_logp = log_probs - wt_log[np.newaxis, :]   # (20, N)

    return delta_logp, wt_idx


def compute_entropy_profile(prob_matrix):
    """
    H(i) = -sum_a P(a|i) log P(a|i)
    
    Returns:
        entropy : np.array (N_windows,) -- one entropy value per window
    """
    entropy = -(prob_matrix * np.log(prob_matrix + 1e-9)).sum(axis=0)  # (N,)
    return entropy


def find_breakers(delta_logp, threshold=-5.0, percentile=None):
    """
    Residues with DeltaLogP < threshold are considered fibril breakers.

    If `percentile` is provided (e.g. 5), the threshold is computed as
    the lowest percentile of the DeltaLogP distribution, ignoring the WT
    diagonal (DeltaLogP = 0 by definition). This is robust to the
    model's score scale and always yields the worst 5% of mutations.

    Args:
        delta_logp : np.array (20, N_windows)
        threshold  : fixed threshold (used if percentile is None)
        percentile : if provided, overrides threshold with the
                     corresponding percentile of the distribution
                     (default None -> use threshold)

    Returns:
        breakers          : list of (window_idx, aa_letter, delta_logp_value)
                            sorted by increasing delta_logp (worst first)
        effective_threshold : float -- the threshold actually used
    """
    if percentile is not None:
        # Excludes exactly the WT-diagonal zeros (WT DeltaLogP = 0 by
        # definition). Does NOT exclude near-zero neutral mutations:
        # those are still informative.
        all_scores = delta_logp.flatten()
        nonzero    = all_scores[all_scores != 0.0]
        effective_threshold = float(np.percentile(nonzero, percentile))
        print(f"  Breaker threshold (p{percentile}): {effective_threshold:.3f}")
    else:
        effective_threshold = threshold
        print(f"  Breaker threshold (fixed): {effective_threshold:.3f}")

    breakers = []
    for win_idx in range(delta_logp.shape[1]):
        for aa_idx in range(20):
            score = delta_logp[aa_idx, win_idx]
            if score < effective_threshold:
                breakers.append((win_idx, AA_ORDER[aa_idx], float(score)))

    breakers.sort(key=lambda x: x[2])
    return breakers, effective_threshold


def find_designable_windows(delta_logp, threshold=-1.0, min_tolerated=5):
    """
    Windows where at least `min_tolerated` mutations have DeltaLogP >
    threshold. Design positions: you can substitute here without
    destabilizing the fibril.
    
    Returns:
        list of window indices
    """
    designable = []
    for win_idx in range(delta_logp.shape[1]):
        n_tolerated = np.sum(delta_logp[:, win_idx] > threshold)
        if n_tolerated >= min_tolerated:
            designable.append(win_idx)
    return designable


def compute_core_score(delta_logp):
    """
    Core score = mean of |DeltaLogP| across all mutations, per window.
    High core score -> highly conserved position (fibril core).
    
    Returns:
        core_score : np.array (N_windows,)
    """
    return np.abs(delta_logp).mean(axis=0)  # (N,)


def compute_mutational_tolerance(delta_logp, threshold=-2.0):
    """
    Number of mutations with DeltaLogP > threshold, per window.
    
    Returns:
        tolerance : np.array (N_windows,) of int
    """
    return (delta_logp > threshold).sum(axis=0)


# ============================================================================
# BLOCK 3 - Embedding extraction + UMAP/PCA
# ============================================================================

@torch.no_grad()
def extract_embedding(model, data, global_idx, device, mask_token_aa, mask_token_cl):
    """
    Extracts the TRUE structural embedding from the model: the vector h
    produced by the BiLSTM+attention, BEFORE the classification head.

    3-tier strategy (from most to least informative):

    Tier 1 - return_embedding=True (TeacherModel/FibrilBreakerModel):
        output['h'] -> vector (d_hidden*2,) -- the real structural embedding.
        The richest option: includes LSTM context + attention.

    Tier 2 - PyTorch hook on the pre-head 'dropout' layer:
        Attaches a forward hook on the last dropout before head_aa.
        Captures h_dropped (d_hidden*2,) without modifying the model.
        Works on a standard ProteinMLM_v2 without return_embedding.

    Tier 3 - logit-space fallback:
        If everything else fails, uses softmax(logits) as a proxy.
        This is NOT a true structural embedding (it's just the AA
        distribution), and a warning is printed.

    Returns:
        h_vec        : np.array (D,)
        is_true_emb  : bool -- True if tier 1 or tier 2 was used
    """
    win_size = data['aa_ids'].shape[1]
    center   = win_size // 2

    aa   = torch.tensor(data['aa_ids'][global_idx],      dtype=torch.long).unsqueeze(0).to(device)
    cl   = torch.tensor(data['cluster_idx'][global_idx],  dtype=torch.long).unsqueeze(0).to(device)
    ss   = torch.tensor(data['ss'][global_idx],           dtype=torch.long).unsqueeze(0).to(device)
    pos  = torch.tensor(data['pos_idx'][global_idx],      dtype=torch.long).unsqueeze(0).to(device)
    sasa = torch.tensor(data['sasa'][global_idx],         dtype=torch.float).unsqueeze(0).to(device)

    aa[:, center] = mask_token_aa
    cl[:, center] = mask_token_cl

    batch = {
        'aa': aa, 'cl': cl, 'ss': ss, 'pos': pos, 'sasa': sasa,
        'y_aa': torch.zeros(1, dtype=torch.long).to(device),
        'y_cl': torch.zeros(1, dtype=torch.long).to(device),
    }

    # -- Tier 1: return_embedding API (Teacher / FibrilBreaker) -------------
    try:
        output = model(batch, return_embedding=True)
        if isinstance(output, dict) and 'h' in output and output['h'] is not None:
            h = output['h'][0].cpu().numpy()
            return h, True
    except Exception:
        pass

    # -- Tier 2: forward hook on the input of head_aa (pre-head) -----------
    # Robust strategy: instead of assuming model.dropout exists (not
    # guaranteed), attach the hook directly to the INPUT of the first
    # Linear layer of head_aa. head_aa is always present in ProteinMLM_v2,
    # so this is guaranteed to work.
    # inp[0] at the input of the first Linear is exactly h_dropped
    # (d_hidden*2,).
    captured = {}

    def _hook(module, inp, out):
        captured['h'] = inp[0].detach()

    hook_handle = None
    try:
        # Find the first Linear layer inside head_aa -- always present in
        # ProteinMLM_v2. Alternatively, if model.dropout exists we use it
        # (cleaner), otherwise we use head_aa[0] (the first Linear).
        target_module = None

        # First choice: an explicit self.dropout (standard ProteinMLM_v2)
        if hasattr(model, 'dropout') and isinstance(model.dropout, nn.Dropout):
            target_module = model.dropout
            hook_on_input = True   # we want inp[0] = h before dropout

        # Second choice: first Linear of head_aa (always present)
        if target_module is None:
            for module in model.head_aa.modules():
                if isinstance(module, nn.Linear):
                    target_module = module
                    hook_on_input = True   # inp[0] = h_dropped
                    break

        if target_module is not None:
            hook_handle = target_module.register_forward_hook(_hook)
            model.eval()
            _ = model(batch)

            if 'h' in captured:
                h = captured['h'][0].cpu().numpy()
                hook_handle.remove()
                return h, True

    except Exception:
        pass
    finally:
        if hook_handle is not None:
            try:
                hook_handle.remove()
            except Exception:
                pass

    # -- Tier 3: logit-space fallback (NOT a true embedding) ----------------
    print("  [WARN] FALLBACK: using the AA distribution as an embedding proxy "
          "(this is not a true structural embedding -- clustering will be less informative).")
    output = model(batch)
    if isinstance(output, (tuple, list)):
        logits = output[0]
    elif isinstance(output, dict):
        logits = output['logits_aa']
    else:
        logits = output
    h = torch.softmax(logits[0], dim=0).cpu().numpy()
    return h, False


def extract_all_embeddings(model, data, matched_indices, device,
                           mask_token_aa=20, mask_token_cl=0):
    """
    Collects embeddings for every matched window.

    Returns:
        embeddings    : np.array (N, D)
        is_true_emb   : bool -- True if every embedding is a true
            structural embedding
    """
    embeddings    = []
    true_emb_flags = []
    print(f"\nExtracting embeddings for {len(matched_indices)} windows...")

    for i, idx in enumerate(matched_indices):
        h, is_true = extract_embedding(model, data, idx, device,
                                       mask_token_aa, mask_token_cl)
        embeddings.append(h)
        true_emb_flags.append(is_true)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(matched_indices)}] [OK]")

    is_true_emb = all(true_emb_flags)
    quality     = "structural (LSTM pre-head)" if is_true_emb else "logit-space proxy"
    print(f"  Embedding quality: {quality}")
    return np.array(embeddings), is_true_emb


def reduce_embeddings(embeddings, method='pca'):
    """
    Reduces embeddings to 2D via PCA (always available) or UMAP (optional).
    
    Returns:
        coords_2d : np.array (N, 2)
        method_used : str
    """
    if method == 'umap':
        try:
            import umap
            reducer     = umap.UMAP(n_components=2, random_state=42)
            coords_2d   = reducer.fit_transform(embeddings)
            return coords_2d, 'UMAP'
        except ImportError:
            print("  [WARN] umap-learn not installed, falling back to PCA.")

    from sklearn.decomposition import PCA
    pca       = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(embeddings)
    return coords_2d, f'PCA (explained var.: {pca.explained_variance_ratio_.sum():.1%})'


# ============================================================================
# BLOCK 6 - Sequence scan: aggregation mutability landscape
# ============================================================================

@torch.no_grad()
def predict_probs_from_window_data(model, data, global_idx, device,
                                   mask_token_aa, mask_token_cl):
    """
    Identical to predict_probs_single -- returns probs (20,) for one
    window. Kept as a separate function for clarity within the
    sequence_scan pipeline.
    """
    return predict_probs_single(model, data, global_idx, device,
                                mask_token_aa, mask_token_cl)


def sequence_scan(model, data, all_protein_indices, fasta_seq, device,
                  mask_token_aa=20, mask_token_cl=0, start_residue=1, end_residue=None,
                  residue_offset=0):
    """
    Scans EVERY position of the protein by masking each corresponding
    window.

    For every position in the FASTA sequence:
      - find the dataset windows whose central residue falls at that position
      - compute the average probability across all available structures
      - compute DeltaLogP, entropy, core_score for that position

    This produces the aggregation mutability landscape across the whole
    sequence: no longer for a single fragment (VGGAV), but for every
    residue of the protein.

    Args:
        model               : loaded ProteinMLM_v2
        data                : NPZ dataset dict
        all_protein_indices : list of every dataset index for this protein
        fasta_seq           : full sequence string (e.g. alpha-syn, 140 aa)
        device              : torch.device

    Returns:
        scan_results : dict with keys:
            'positions'    : list[int] -- 1-based positions with at least 1 window
            'wt_aas'       : list[str] -- WT AA at each position
            'mean_probs'   : np.array (L_covered, 20) -- mean probability
            'delta_logp'   : np.array (L_covered, 20) -- DeltaLogP per position
            'entropy'      : np.array (L_covered,)
            'core_score'   : np.array (L_covered,)
            'tolerance'    : np.array (L_covered,) -- n tolerated mutations
            'n_structures' : list[int] -- n structures per position
    """
    aa_ids   = data['aa_ids']
    win_size = aa_ids.shape[1]
    center   = win_size // 2

    # Map: 0-based FASTA position -> list of global_idx in the dataset
    # FIX: find ALL occurrences of the window in the FASTA, not just the
    # first one. alpha-syn contains sequence repeats -> a window can map
    # to multiple positions.
    pos_to_indices = defaultdict(list)

    for idx in all_protein_indices:
        win_seq = window_to_seq(aa_ids[idx])
        # Iterate over every occurrence in the FASTA
        start = 0
        while True:
            fasta_pos = fasta_seq.find(win_seq, start)
            if fasta_pos == -1:
                break
            center_pos = fasta_pos + center  # 0-based position of the central residue
            pos_to_indices[center_pos].append(idx)
            start = fasta_pos + 1  # advance by 1 to find overlapping occurrences

    if not pos_to_indices:
        print("  [WARN] Sequence scan: no window mapped onto the FASTA.")
        return None

    # Filter positions before start_residue (in canonical numbering)
    if start_residue > 1:
        min_pos_0based = start_residue - 1 - residue_offset
        filtered_keys = [p for p in pos_to_indices if p >= min_pos_0based]
        n_removed = len(pos_to_indices) - len(filtered_keys)
        if n_removed > 0:
            print(f"  [start_residue={start_residue}] Excluded {n_removed} positions "
                  f"before residue {start_residue}.")
        pos_to_indices = {k: pos_to_indices[k] for k in filtered_keys}

    # Filter positions after end_residue (in canonical numbering)
    if end_residue is not None:
        max_pos_0based = end_residue - 1 - residue_offset
        filtered_keys = [p for p in pos_to_indices if p <= max_pos_0based]
        n_removed = len(pos_to_indices) - len(filtered_keys)
        if n_removed > 0:
            print(f"  [end_residue={end_residue}] Excluded {n_removed} positions "
                  f"after residue {end_residue}.")
        pos_to_indices = {k: pos_to_indices[k] for k in filtered_keys}

    sorted_positions = sorted(pos_to_indices.keys())
    print(f"\n[SEQUENCE SCAN] Positions covered: {len(sorted_positions)} / {len(fasta_seq)}")
    print(f"  Total windows to process: "
          f"{sum(len(v) for v in pos_to_indices.values())}")

    # -- Pre-compute probs for ALL unique indices in a single pass ----------
    # Instead of calling predict_probs for each (position, structure) pair
    # separately, we compute all probabilities for every unique index just
    # once. This reduces the number of forward passes from
    # O(N_windows) to O(N_unique_indices).
    all_unique_indices = sorted(set(
        idx for indices in pos_to_indices.values() for idx in indices
    ))
    print(f"  Unique indices (forward passes): {len(all_unique_indices)}")

    idx_to_probs = {}
    for i, idx in enumerate(all_unique_indices):
        p = predict_probs_from_window_data(model, data, idx, device,
                                           mask_token_aa, mask_token_cl)
        idx_to_probs[idx] = p
        if (i + 1) % 50 == 0 or i == len(all_unique_indices) - 1:
            print(f"  [{i+1}/{len(all_unique_indices)}] forward passes [OK]")

    positions    = []
    wt_aas       = []
    mean_probs   = []
    n_structures = []

    for pos_0 in sorted_positions:
        indices = pos_to_indices[pos_0]
        wt_aa   = fasta_seq[pos_0]

        if wt_aa not in AA_TO_IDX:
            continue  # skip non-standard residues

        # Average probability: uses the cache, no extra forward pass
        probs_list = [idx_to_probs[idx] for idx in indices if idx in idx_to_probs]
        if not probs_list:
            continue

        mean_p = np.mean(probs_list, axis=0)  # (20,)

        positions.append(pos_0 + 1 + residue_offset)  # canonical numbering
        wt_aas.append(wt_aa)
        mean_probs.append(mean_p)
        n_structures.append(len(probs_list))

    mean_probs_arr = np.array(mean_probs)   # (L, 20)

    # DeltaLogP per position (WT specific to that position)
    log_probs = np.log(mean_probs_arr + 1e-9)   # (L, 20)
    delta_logp_scan = np.zeros_like(log_probs)
    for i, wt_aa in enumerate(wt_aas):
        wt_idx_i = AA_TO_IDX[wt_aa]
        delta_logp_scan[i] = log_probs[i] - log_probs[i, wt_idx_i]

    # Metrics
    entropy_scan    = -(mean_probs_arr * np.log(mean_probs_arr + 1e-9)).sum(axis=1)
    core_score_scan = np.abs(delta_logp_scan).mean(axis=1)
    tolerance_scan  = (delta_logp_scan > -2.0).sum(axis=1)

    # -- SIGNED average (not absolute value) --------------------------------
    # Distinguishes two phenomena that core_score (absolute value)
    # conflates:
    #   mean_signed < 0  -> the WT is (near) optimal, mutations are
    #                      penalized -> a true conserved hotspot/breaker
    #   mean_signed >= 0 -> the WT itself looks sub-optimal according to
    #                      the model, almost every substitution improves
    #                      it -> a "tolerant" position, not an aggregation
    #                      hotspot in the classic sense
    mean_signed_scan = delta_logp_scan.mean(axis=1)

    print(f"  Positions with data: {len(positions)}")
    print(f"  Structures per position: min={min(n_structures)}, "
          f"max={max(n_structures)}, mean={np.mean(n_structures):.1f}")

    return {
        'positions':    positions,
        'wt_aas':       wt_aas,
        'mean_probs':   mean_probs_arr,
        'delta_logp':   delta_logp_scan,
        'entropy':      entropy_scan,
        'core_score':   core_score_scan,
        'mean_signed':  mean_signed_scan,
        'tolerance':    tolerance_scan,
        'n_structures': n_structures,
    }


def find_sensitivity_threshold(values, n_bins=50, smooth_sigma=1.5,
                               zero_fraction=0.05, fallback_percentile=75):
    """
    Finds the threshold for the Hotspot Index as the point where the
    distribution's density first drops close to zero after the dominant
    peak.

    Biological motivation: the distribution has a single large peak
    (flexible/tolerant positions) followed by a tail that flattens out
    around a certain value. Positions to the right of that point are the
    rigid/sensitive positions (the tail's outliers).

    Algorithm:
      1. Estimate the density with a histogram + Gaussian smoothing
      2. Find the dominant peak (maximum density)
      3. After the dominant peak, scan right until the first bin where
         smoothed_density < zero_fraction * peak_density
         (default: 5% of the peak density = "near zero")
      4. That bin is the threshold

    Args:
        values            : 1D array of hotspot index values
        n_bins            : number of histogram bins (default 50)
        smooth_sigma      : sigma of the Gaussian filter (default 1.5)
        zero_fraction     : fraction of the peak density considered "zero"
                            (default 0.05 = 5%)
        fallback_percentile: percentile used if the algorithm does not converge

    Returns:
        threshold : float -- the optimal threshold
        info      : dict with keys:
                      'method'           : 'density_near_zero' or 'percentile_fallback'
                      'peaks'            : [x of the dominant peak]
                      'valley'           : threshold found (float or None)
                      'zero_level'       : density considered "zero" (5% of the peak)
                      'bin_centers'      : histogram x array
                      'smoothed_density' : smoothed density array
    """
    values = np.asarray(values, dtype=float)

    # -- 1. Histogram + smoothing ---------------------------------------------
    counts, bin_edges = np.histogram(values, bins=n_bins, density=True)
    bin_centers       = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    smoothed          = gaussian_filter1d(counts, sigma=smooth_sigma)

    # -- 2. Dominant peak -----------------------------------------------------
    peak_idxs, _ = find_peaks(smoothed)

    info_base = {
        'bin_centers':       bin_centers,
        'smoothed_density':  smoothed,
    }

    if len(peak_idxs) == 0:
        thr = float(np.percentile(values, fallback_percentile))
        print(f"  [WARN] No peak found. Fallback: p{fallback_percentile} = {thr:.4f}")
        return thr, {**info_base, 'method': f'percentile_fallback_p{fallback_percentile}',
                     'peaks': [], 'valley': None, 'zero_level': None}

    dominant_idx      = int(peak_idxs[np.argmax(smoothed[peak_idxs])])
    peak_density      = float(smoothed[dominant_idx])
    zero_level        = zero_fraction * peak_density

    # -- 3. First drop below zero_level after the dominant peak --------------
    region_after = smoothed[dominant_idx:]
    below_idxs   = np.where(region_after < zero_level)[0]

    if len(below_idxs) == 0:
        # The tail never drops below the threshold: fall back to a percentile
        thr = float(np.percentile(values, fallback_percentile))
        print(f"  [WARN] The density never drops below {zero_fraction*100:.0f}% of the peak. "
              f"Fallback: p{fallback_percentile} = {thr:.4f}")
        return thr, {**info_base, 'method': f'percentile_fallback_p{fallback_percentile}',
                     'peaks': [float(bin_centers[dominant_idx])],
                     'valley': None, 'zero_level': zero_level}

    threshold_bin_idx = dominant_idx + int(below_idxs[0])
    threshold         = float(bin_centers[threshold_bin_idx])

    print(f"  Dominant peak    : {bin_centers[dominant_idx]:.4f}  "
          f"(density {peak_density:.4f})")
    print(f"  'Zero' level     : {zero_level:.4f}  ({zero_fraction*100:.0f}% of the peak)")
    print(f"  [OK] Threshold (first drop to zero): {threshold:.4f}")

    return threshold, {**info_base,
                       'method':     'density_near_zero',
                       'peaks':      [float(bin_centers[dominant_idx])],
                       'valley':     threshold,
                       'zero_level': zero_level}



def _draw_sensitivity_distribution(ax, core_score, threshold, bimodal_info):
    """
    Draws the distribution of Hotspot Index values on an existing axis
    (used both in the main figure and in the standalone plot).

    Layout:
      - Bicolor-filled histogram (< threshold = light purple, > threshold = dark purple)
      - Smoothed KDE-like curve
      - Vertical lines at the peaks (dashed green)
      - Vertical line at the threshold (dashed orange) + shading
      - Text annotation of the threshold value
    """
    bin_centers = bimodal_info['bin_centers']
    smoothed    = bimodal_info['smoothed_density']
    n_bins      = len(bin_centers)
    is_bimodal  = bimodal_info.get('valley') is not None
    zero_level  = bimodal_info.get('zero_level')
    method      = bimodal_info.get('method', '')

    # Palette consistent with the DeltaLogP heatmap above (RdBu, red side):
    COLOR_RIGID    = '#67001F'   # darkest RdBu red (residue 42 in the heatmap)
    COLOR_FLEXIBLE = '#FDDBC7'   # pale pink RdBu (Y/A53 in the heatmap)
    COLOR_CURVE    = '#4A0015'   # even darker, for the KDE curve
    COLOR_THR      = '#111111'   # black for the threshold dashed line

    # -- Bicolor histogram ---------------------------------------------------
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    for cx, dens in zip(bin_centers, smoothed):
        color = COLOR_RIGID if cx > threshold else COLOR_FLEXIBLE
        ax.bar(cx, dens, width=bin_width * 0.92, color=color, alpha=0.55,
               edgecolor='none', align='center')

    # -- Smoothed KDE-like curve ----------------------------------------------
    ax.plot(bin_centers, smoothed, color=COLOR_CURVE, lw=2.2,
            label='Smoothed density', zorder=4)

    # -- Horizontal "near-zero" level line ------------------------------------
    if zero_level is not None:
        ax.axhline(zero_level, color='#888', linestyle=':', lw=1.2, alpha=0.7,
                   label=f'threshold ({zero_level:.3f})')

    # -- Dominant peak ----------------------------------------------------------
    peaks = bimodal_info.get('peaks', [])
    if peaks:
        ax.axvline(peaks[0], color='#555', linestyle=':', lw=1.4, zorder=3)
        peak_y = float(np.interp(peaks[0], bin_centers, smoothed))
        ax.annotate(
            'dominant\npeak',
            xy=(peaks[0], peak_y),
            xytext=(peaks[0], peak_y + 0.06 * smoothed.max()),
            ha='center', va='bottom', fontsize=9, color='#333', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.0),
        )

    # -- Threshold: black line + shading + annotation -------------------------
    y_max = smoothed.max() * 1.35
    thr_label = (
        f'threshold = {threshold:.3f}'
        if method == 'density_near_zero'
        else f'p75 fallback = {threshold:.3f}'
    )
    ax.axvline(threshold, color=COLOR_THR, linestyle='--', lw=2.0, zorder=5,
               label=thr_label)

    ax.fill_between(bin_centers, smoothed,
                    where=(bin_centers <= threshold),
                    color=COLOR_FLEXIBLE, alpha=0.15, label='flexible positions')
    ax.fill_between(bin_centers, smoothed,
                    where=(bin_centers >= threshold),
                    color=COLOR_RIGID, alpha=0.15, label='rigid/sensitive positions')

    ax.annotate(
        f'threshold\n{threshold:.3f}',
        xy=(threshold, smoothed.max() * 0.45),
        xytext=(threshold + (bin_centers[-1] - bin_centers[0]) * 0.06,
                smoothed.max() * 0.65),
        ha='left', va='center', fontsize=9, color=COLOR_THR, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLOR_THR, lw=1.3),
        zorder=6,
    )

    ax.set_ylim(0, y_max)
    ax.set_xlabel('Hotspot Index (mean |DeltaLogP|)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14)
    method_str = 'near-zero cutoff' if method == 'density_near_zero' else 'p75 fallback'
    ax.set_title(f'Hotspot Index distribution  -  threshold: {method_str}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.7)
    ax.tick_params(labelsize=14)


def plot_sensitivity_distribution_standalone(core_score, threshold, bimodal_info,
                                             protein_name='Protein', output_path=None):
    """
    Standalone figure of the Hotspot Index value distribution with the
    bimodal threshold highlighted.

    Saved as 'hotspot_index_distribution.png' in the output folder.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='white')
    _draw_sensitivity_distribution(ax, core_score, threshold, bimodal_info)
    fig.suptitle(
        f'{protein_name} - Hotspot Index distribution',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if output_path:
        _save_multiformat(fig, output_path, label="hotspot_distribution")
    plt.close(fig)


def plot_aggregation_mutability_landscape(scan_results, protein_name,
                                          output_path=None, figsize=(22, 16)):
    """
    Main figure for the paper: aggregation mutability landscape.

    3 stacked panels:
      1. DeltaLogP heatmap (20 x L_covered)  -- full landscape
      2. Hotspot Index bar chart     -- mean |DeltaLogP| per position
      3. Hotspot Index distribution  -- histogram + KDE + bimodal threshold
    """
    positions   = scan_results['positions']
    wt_aas      = scan_results['wt_aas']
    delta_logp  = scan_results['delta_logp']    # (L, 20)
    core_score  = scan_results['core_score']     # (L,)

    L = len(positions)
    x = np.arange(L)
    tick_positions = list(range(0, L, 3))

    # -- Compute the bimodal threshold only once -----------------------------
    core_thr, bimodal_info = find_sensitivity_threshold(core_score)
    method_label = (
        f"threshold = {core_thr:.3f}"
        if bimodal_info['method'] == 'density_near_zero'
        else f"p{75} fallback = {core_thr:.3f}"
    )

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs  = gridspec.GridSpec(3, 1, hspace=0.50, height_ratios=[3, 1, 1.4], figure=fig)

    # -- Panel 1: DeltaLogP heatmap --------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    delta_T = delta_logp.T   # (20, L) for imshow
    vmax    = max(2.0, float(np.percentile(np.abs(delta_T), 97)))
    norm    = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax1.imshow(delta_T, aspect='auto', cmap=plt.cm.RdBu, norm=norm,
                    interpolation='nearest', extent=[-0.5, L - 0.5, 19.5, -0.5])

    for i, wt_aa in enumerate(wt_aas):
        if wt_aa in AA_TO_IDX:
            ax1.plot(i, AA_TO_IDX[wt_aa], 'k.', markersize=8, alpha=0.7)

    ax1.set_yticks(range(20))
    ax1.set_yticklabels([AA_NAMES[i] for i in range(20)], fontsize=14)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(
        [f"{wt_aas[i]}{positions[i]}" for i in tick_positions],
        rotation=45, ha='right', rotation_mode='anchor', fontsize=14
    )
    ax1.set_ylabel('Amino acid', fontsize=14, fontweight='bold')
    ax1.set_title(
        f'Mutability Landscape - {protein_name}\n'
        f'DeltaLogP for each position in the sequence (black dot = WT)',
        fontsize=13, fontweight='bold', pad=12
    )

    cbar = plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.01)
    cbar.set_label('DeltaLogP', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # -- Panel 2: Hotspot Index bar chart --------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # mean_signed distinguishes the direction of the average shift:
    #   < 0  -> WT (near) optimal, mutations penalized -> a true hotspot/breaker
    #   >= 0 -> WT sub-optimal according to the model, mutations favored ->
    #          a "tolerant" position, high core_score but NOT an
    #          aggregation hotspot
    mean_signed = scan_results.get('mean_signed', delta_logp.mean(axis=1))

    COLOR_BREAKER  = "#9B1111"   # dark red - true hotspot (WT optimal, mutation = worse)
    COLOR_TOLERANT = "#1B5E6B"   # teal - WT sub-optimal (mutation = better)
    COLOR_NEUTRAL  = "#EEC2A9"   # light peach - flexible, not sensitive position

    colors_bar = []
    for v, ms in zip(core_score, mean_signed):
        if v <= core_thr:
            colors_bar.append(COLOR_NEUTRAL)
        elif ms < 0:
            colors_bar.append(COLOR_BREAKER)
        else:
            colors_bar.append(COLOR_TOLERANT)

    ax2.bar(x, core_score, color=colors_bar, alpha=0.80, edgecolor='none')
    ax2.axhline(core_thr, color='#111111', linestyle='--', lw=1.5, alpha=0.9)

    legend_handles = [
        Line2D([0], [0], color='#111111', linestyle='--', lw=1.5, label=method_label),
    ]
    ax2.legend(handles=legend_handles, fontsize=9, loc='upper right')
    ax2.set_ylabel('Hotspot\nIndex', fontsize=14, fontweight='bold')
    ax2.tick_params(labelsize=14)

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(
        [f"{wt_aas[i]}{positions[i]}" for i in tick_positions],
        rotation=45, ha='right', rotation_mode='anchor', fontsize=14
    )
    ax2.set_xlabel('Sequence position', fontsize=14, fontweight='bold', labelpad=8)

    # -- Panel 3: Hotspot Index distribution -----------------------
    # Does not share the X axis with the panels above (different axis: values, not positions)
    ax3 = fig.add_subplot(gs[2])
    _draw_sensitivity_distribution(ax3, core_score, core_thr, bimodal_info)

    plt.suptitle(
        f'{protein_name} - Full Sequence Aggregation Mutability Landscape',
        fontsize=14, fontweight='bold', y=1.01
    )

    if output_path:
        _save_multiformat(fig, output_path, label="aggregation_mutability_landscape")

        # Standalone figure with only the distribution (for paper / supplementary)
        diag_path = str(Path(output_path).parent / 'hotspot_index_distribution.png')
        plot_sensitivity_distribution_standalone(
            core_score, core_thr, bimodal_info,
            protein_name=protein_name, output_path=diag_path
        )
    plt.close(fig)


def save_scan_report(scan_results, protein_name, out_dir):
    """Saves the scan CSV: one row per position, with the top-3 compatible mutations and breakers."""
    rows = []
    delta_logp = scan_results['delta_logp']   # (L, 20)

    for i, (pos, wt_aa) in enumerate(zip(scan_results['positions'],
                                          scan_results['wt_aas'])):
        dlp    = delta_logp[i]
        wt_idx = AA_TO_IDX.get(wt_aa, -1)

        # Top-3 compatible (excludes WT)
        sorted_desc = np.argsort(dlp)[::-1]
        compat = [(AA_ORDER[j], round(float(dlp[j]), 3))
                  for j in sorted_desc if j != wt_idx][:3]

        # Top-3 breakers
        sorted_asc = np.argsort(dlp)
        breakers   = [(AA_ORDER[j], round(float(dlp[j]), 3))
                      for j in sorted_asc if j != wt_idx][:3]

        mean_signed_i = float(scan_results['mean_signed'][i])
        is_core_i     = scan_results['core_score'][i] > np.percentile(scan_results['core_score'], 75)

        # Explicit category: distinguishes "true" hotspots (breaker, WT
        # optimal) from positions where the WT itself looks sub-optimal
        # according to the model (high core_score only because almost
        # every mutation is FAVORABLE).
        if not is_core_i:
            category_i = 'neutral'
        elif mean_signed_i < 0:
            category_i = 'breaker_hotspot'      # true hotspot: WT optimal, mutations penalized
        else:
            category_i = 'tolerant_suboptimal_wt'  # WT itself sub-optimal, mutations favored

        rows.append({
            'position':             pos,
            'wt_aa':                wt_aa,
            'n_structures':         scan_results['n_structures'][i],
            'entropy':              round(float(scan_results['entropy'][i]), 4),
            'core_score':           round(float(scan_results['core_score'][i]), 4),
            'mean_signed_delta_logp': round(mean_signed_i, 4),
            'category':             category_i,
            'n_tolerated':          int(scan_results['tolerance'][i]),
            'is_hotspot':           scan_results['entropy'][i] > np.percentile(scan_results['entropy'], 75),
            'is_core':              is_core_i,
            'top1_compat':          f"{compat[0][0]}({compat[0][1]:.2f})" if compat else '',
            'top2_compat':          f"{compat[1][0]}({compat[1][1]:.2f})" if len(compat) > 1 else '',
            'top3_compat':          f"{compat[2][0]}({compat[2][1]:.2f})" if len(compat) > 2 else '',
            'top1_breaker':         f"{breakers[0][0]}({breakers[0][1]:.2f})" if breakers else '',
            'top2_breaker':         f"{breakers[1][0]}({breakers[1][1]:.2f})" if len(breakers) > 1 else '',
        })

    csv_path = Path(out_dir) / 'sequence_scan_report.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[OK] Sequence scan CSV saved: {csv_path}")
    return csv_path

def plot_delta_logp(delta_logp, column_labels, window_seq, wt_idx,
                   output_path=None, figsize=(18, 10), protein_name='Protein'):
    """
    DeltaLogP heatmap (20 AA x N windows).
    Diverging colormap: blue = compatible, red = breaker.
    """
    row_labels = [AA_NAMES[i] for i in range(20)]

    vmax = max(2.0, float(np.percentile(np.abs(delta_logp), 95)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    im = ax.imshow(delta_logp, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    # Grid
    ax.set_xticks(np.arange(delta_logp.shape[1]) + 0.5, minor=True)
    ax.set_yticks(np.arange(20) + 0.5, minor=True)
    ax.grid(which='minor', color='#cccccc', linestyle='-', linewidth=0.4)
    ax.tick_params(which='minor', size=0)
    ax.set_xlim(-0.5, delta_logp.shape[1] - 0.5)
    ax.set_ylim(20 - 0.5, -0.5)

    # Values inside the cells
    for c in range(delta_logp.shape[1]):
        for r in range(20):
            val = delta_logp[r, c]
            if abs(val) > 0.05:
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                        color=color, fontsize=10, fontweight='bold')
            else:
                # Neutral value (includes the WT row = exact 0.0)
                ax.text(c, r, '0.0', ha='center', va='center',
                        color='#999999', fontsize=9.5, fontstyle='italic')

    # Highlight the WT row
    ax.add_patch(plt.Rectangle(
        (-0.5, wt_idx - 0.5), delta_logp.shape[1], 1,
        fill=False, edgecolor='#FFD700', linewidth=2.5, zorder=3
    ))

    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha='right',
                   rotation_mode='anchor', fontsize=11)
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(
        [f'{"-> " if i == wt_idx else "   "}{AA_NAMES[i]}' for i in range(20)],
        fontsize=14
    )

    ax.set_xlabel(f'{protein_name} windows', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Mutation', fontsize=14, fontweight='bold')
    ax.set_title(
        f'DeltaLogP Mutational Landscape - {protein_name}\n'
        f'Window: {window_seq}  |  WT center: {AA_ORDER[wt_idx]} (highlighted row)',
        fontsize=14, fontweight='bold', pad=15
    )


    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('DeltaLogP  (mut - WT)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()

    if output_path:
        _save_multiformat(fig, output_path, label="heatmap_delta_logp")
    plt.close(fig)


def plot_analysis_summary(entropy, core_score, tolerance, delta_logp,
                          column_labels, window_seq, wt_idx,
                          breaker_threshold=-5.0, output_path=None,
                          figsize=(18, 14)):
    """
    Multi-panel figure (4 rows):
      1. Entropy profile
      2. Core score (fibril conservation)
      3. Mutational tolerance
      4. WT probability vs window (sanity check)
    """
    N    = len(entropy)
    x    = np.arange(N)
    xlbls = [f"W{i+1}" for i in range(N)]

    # Hotspot / core thresholds
    ent_thr   = np.percentile(entropy, 75)
    core_thr  = np.percentile(core_score, 75)

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs  = gridspec.GridSpec(4, 1, hspace=0.55, figure=fig)

    # -- Panel 1: Entropy -------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(x, entropy, color='#4A90D9', edgecolor='none', alpha=0.85)
    for i, (v, b) in enumerate(zip(entropy, bars)):
        if v > ent_thr:
            b.set_color('#4A90D9')
            b.set_alpha(0.95)
    ax1.axhline(ent_thr, color='#4A90D9', linestyle='--', lw=1.5,
                label=f'75th percentile ({ent_thr:.2f})')
    ax1.set_ylabel('Shannon Entropy', fontsize=10, fontweight='bold')
    ax1.set_title(f'Mutational entropy per window - {window_seq}',
                  fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlbls, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_xlim(-0.5, N - 0.5)

    # Hotspot annotations
    for i, v in enumerate(entropy):
        if v > ent_thr:
            ax1.text(i, v + 0.01, 'H', ha='center', va='bottom',
                     fontsize=7, color='#E8512B', fontweight='bold')

    # -- Panel 2: Core score --------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(x, core_score, color='#2E7D32', edgecolor='none', alpha=0.85)
    for i, (v, b) in enumerate(zip(core_score, bars2)):
        if v > core_thr:
            b.set_color('#2E7D32')
    ax2.axhline(core_thr, color='#2E7D32', linestyle='--', lw=1.5,
                label=f'75th percentile ({core_thr:.2f})')
    ax2.set_ylabel('Fibril Core Score\n(mean |DeltaLogP|)', fontsize=10, fontweight='bold')
    ax2.set_title('Structural conservation (high = fibril core)',
                  fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlbls, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_xlim(-0.5, N - 0.5)

    for i, v in enumerate(core_score):
        if v > core_thr:
            ax2.text(i, v + 0.02, 'C', ha='center', va='bottom',
                     fontsize=7, color='#1A237E', fontweight='bold')

    # -- Panel 3: Mutational tolerance -----------------------------------------
    ax3 = fig.add_subplot(gs[2])
    bars3 = ax3.bar(x, tolerance, color='#7B1FA2', edgecolor='none', alpha=0.85)
    for i, (v, b) in enumerate(zip(tolerance, bars3)):
        if v >= 5:
            b.set_color('#FF6F00')
    ax3.axhline(5, color='#FF6F00', linestyle='--', lw=1.5,
                label='Min. 5 tolerated mutations (designable)')
    ax3.set_ylabel('N tolerated mutations\n(DeltaLogP > -2)', fontsize=10, fontweight='bold')
    ax3.set_title('Mutational tolerance - designable positions',
                  fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlbls, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.set_xlim(-0.5, N - 0.5)

    for i, v in enumerate(tolerance):
        ax3.text(i, v + 0.1, str(int(v)), ha='center', va='bottom',
                 fontsize=7, color='black')

    # -- Panel 4: WT probability -------------------------------------------------
    ax4 = fig.add_subplot(gs[3])
    # DeltaLogP of the WT relative to itself = 0, but here we show the
    # absolute probability instead
    wt_delta = delta_logp[wt_idx, :]   # (N,) -- 0 by definition when masked
    # Show instead the average absolute value of breakers per window
    n_breakers_per_win = (delta_logp < breaker_threshold).sum(axis=0)
    ax4.bar(x, n_breakers_per_win, color='#C62828', edgecolor='none', alpha=0.85)
    ax4.set_ylabel(f'N breaker mutations\n(DeltaLogP < {breaker_threshold})',
                   fontsize=10, fontweight='bold')
    ax4.set_title(f'Breaker mutations per window (threshold: {breaker_threshold})',
                  fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(xlbls, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    ax4.set_xlim(-0.5, N - 0.5)

    for i, v in enumerate(n_breakers_per_win):
        if v > 0:
            ax4.text(i, v + 0.05, str(int(v)), ha='center', va='bottom',
                     fontsize=7, color='black')

    if output_path:
        _save_multiformat(fig, output_path, label="analysis_summary")
    plt.close(fig)


def plot_embedding_scatter(coords_2d, method_label, entropy, core_score,
                           column_labels, window_seq, output_path=None,
                           figsize=(14, 6)):
    """
    2D scatter of the structural embeddings, colored by entropy (left)
    and core score (right).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white')

    for ax, values, cmap, label in zip(
        axes,
        [entropy, core_score],
        ['plasma', 'viridis'],
        ['Shannon Entropy', 'Fibril Core Score']
    ):
        sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                        c=values, cmap=cmap, s=80, alpha=0.85, edgecolors='k', lw=0.5)
        for i, lbl in enumerate(column_labels):
            short = lbl.split('\n')[1] if '\n' in lbl else lbl
            ax.annotate(short, (coords_2d[i, 0], coords_2d[i, 1]),
                        textcoords='offset points', xytext=(5, 4),
                        fontsize=7, alpha=0.8)
        plt.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)
        ax.set_xlabel(f'{method_label} 1', fontsize=10)
        ax.set_ylabel(f'{method_label} 2', fontsize=10)
        ax.set_title(f'Embedding clustering\n{label}', fontsize=11, fontweight='bold')

    fig.suptitle(
        f'Structural embedding space - {window_seq}  [{method_label}]',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    if output_path:
        _save_multiformat(fig, output_path, label="embedding_scatter")
    plt.close(fig)


# ============================================================================
# BLOCK 5 - JSON / CSV report
# ============================================================================

def build_report(delta_logp, entropy, core_score, tolerance,
                 column_labels, window_seq, wt_idx,
                 breaker_threshold=-5.0, design_threshold=-1.0):
    """
    Builds a dict with the full numeric summary.
    """
    center = len(window_seq) // 2
    wt_aa  = window_seq[center]

    breakers, _  = find_breakers(delta_logp, threshold=breaker_threshold, percentile=None)
    designable = find_designable_windows(delta_logp, design_threshold)
    hotspots   = np.where(entropy > np.percentile(entropy, 75))[0].tolist()
    core_wins  = np.where(core_score > np.percentile(core_score, 75))[0].tolist()

    # For each window, top-3 most compatible mutations and top-3 breakers
    per_window = []
    for win_idx, lbl in enumerate(column_labels):
        dlp = delta_logp[:, win_idx]
        top3_compat  = [(AA_ORDER[i], float(dlp[i]))
                        for i in np.argsort(dlp)[::-1][:3] if i != wt_idx]
        top3_breaker = [(AA_ORDER[i], float(dlp[i]))
                        for i in np.argsort(dlp)[:3] if i != wt_idx]
        per_window.append({
            'window_idx':    win_idx,
            'label':         lbl.replace('\n', '_'),
            'entropy':       float(entropy[win_idx]),
            'core_score':    float(core_score[win_idx]),
            'n_tolerated':   int(tolerance[win_idx]),
            'is_hotspot':    win_idx in hotspots,
            'is_core':       win_idx in core_wins,
            'is_designable': win_idx in designable,
            'top3_compatible': top3_compat,
            'top3_breaker':    top3_breaker,
        })

    report = {
        'protein':   'alpha-synuclein',
        'window':    window_seq,
        'wt_center': wt_aa,
        'n_windows': int(delta_logp.shape[1]),
        'breaker_threshold':  breaker_threshold,
        'design_threshold':   design_threshold,
        'n_breaker_pairs':    len(breakers),
        'n_designable_windows': len(designable),
        'n_hotspot_windows':    len(hotspots),
        'n_core_windows':       len(core_wins),
        'global_entropy_mean':  float(entropy.mean()),
        'global_core_mean':     float(core_score.mean()),
        'worst_breakers':       breakers[:10],
        'per_window':           per_window,
    }
    return report


def save_report(report, out_dir):
    """Saves the report as JSON and CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / 'analysis_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[OK] JSON report saved: {json_path}")

    # CSV (per-window)
    rows = []
    for w in report['per_window']:
        rows.append({
            'window_idx':     w['window_idx'],
            'label':          w['label'],
            'entropy':        round(w['entropy'], 4),
            'core_score':     round(w['core_score'], 4),
            'n_tolerated':    w['n_tolerated'],
            'is_hotspot':     w['is_hotspot'],
            'is_core':        w['is_core'],
            'is_designable':  w['is_designable'],
            'top1_compat_aa': w['top3_compatible'][0][0] if w['top3_compatible'] else '',
            'top1_compat_dlp': w['top3_compatible'][0][1] if w['top3_compatible'] else '',
            'top1_breaker_aa': w['top3_breaker'][0][0] if w['top3_breaker'] else '',
            'top1_breaker_dlp': w['top3_breaker'][0][1] if w['top3_breaker'] else '',
        })
    csv_path = out_dir / 'analysis_report.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[OK] CSV report saved: {csv_path}")

    # Print the worst breakers
    print("\n-- Top-10 breaker mutations --")
    print(f"{'Win':>5} {'AA':>4} {'DeltaLogP':>10}")
    print("-" * 25)
    for win_idx, aa, score in report['worst_breakers']:
        print(f"  {win_idx:>3}   {aa:>2}   {score:>9.3f}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Probability heatmap + full mutational analysis for a given protein"
    )
    # Input
    ap.add_argument('--model', required=True,
                     help="Path to the trained model checkpoint (.pt file, e.g. teachers/teacher_X/best_teacher.pt).")
    ap.add_argument('--data', required=True,
                     help="Path to the windows dataset (.npz file, e.g. windows_all_local.npz).")
    ap.add_argument('--protein_metadata', required=True,
                     help="Path to the JSON file mapping window index to {pdb_id, protein_class}.")
    ap.add_argument('--rotamer_csv', default='rotamer_localmax_maxdens.csv',
                     help="Path to the rotamer library CSV (default: rotamer_localmax_maxdens.csv).")
    ap.add_argument('--fasta', default=None,
                     help="Path to a FASTA file with the full protein sequence. Required for --sequence_scan and for showing sequence positions in labels.")
    ap.add_argument('--window_seq', default='NVGGA',
                     help="5-letter amino acid window sequence to analyze (e.g. 'VGGAV') (default: NVGGA).")
    ap.add_argument('--protein_name', default='alpha-synuclein',
                     help="Protein class name to filter windows by, and to display in plot titles (default: alpha-synuclein).")

    # Tokens
    ap.add_argument('--mask_token_aa', type=int, default=20,
                     help="Integer index used as the MASK token for the amino acid channel (default: 20).")
    ap.add_argument('--mask_token_cl', type=int, default=0,
                     help="Integer index used as the MASK token for the cluster channel (default: 0).")

    # Output
    ap.add_argument('--output',   default='results/asynuclein_analysis',
                    help='Output folder (created if it does not exist)')

    # Analysis
    ap.add_argument('--breaker_threshold', type=float, default=None,
                    help='FIXED DeltaLogP threshold to define breakers. '
                         'If omitted, uses --breaker_percentile (recommended).')
    ap.add_argument('--breaker_percentile', type=float, default=5.0,
                    help='Percentile for the adaptive breaker threshold (default: 5 = bottom 5%%). '
                         'Ignored if --breaker_threshold is provided.')
    ap.add_argument('--design_threshold',  type=float, default=-1.0,
                    help='DeltaLogP threshold to define tolerated mutations (default: -1)')
    ap.add_argument('--min_tolerated',     type=int,   default=5,
                    help='Min tolerated mutations per designable window (default: 5)')
    ap.add_argument('--embedding_method',  default='pca',
                    choices=['pca', 'umap'],
                    help='Embedding reduction method (default: pca)')
    ap.add_argument('--skip_embedding',    action='store_true',
                    help='Skip embedding extraction (faster)')
    ap.add_argument('--sequence_scan',     action='store_true',
                    help='Run a scan over the WHOLE sequence (requires --fasta)')
    ap.add_argument('--start_residue',    type=int, default=1,
                    help='Starting residue in CANONICAL numbering (default: 1 = no filter). '
                         'Interpreted according to --residue_offset. '
                         'Windows and positions before this residue are excluded.')
    ap.add_argument('--end_residue',      type=int, default=None,
                    help='Final residue in CANONICAL numbering (default: None = no filter). '
                         'Interpreted according to --residue_offset. '
                         'Windows and positions after this residue are excluded.')
    ap.add_argument('--residue_offset',   type=int, default=0,
                    help='Offset used to convert the 1-based FASTA position into canonical '
                         'residue numbering: canonical_residue = FASTA_position + offset. '
                         'Needed when the construct FASTA includes an added Met/His-tag/linker. '
                         'E.g.: HET-s 2RNM -> --residue_offset 216 (FASTA pos 2 = K218); '
                         'Mrip3/RIPK3 6JPD -> --residue_offset 402 (FASTA pos 7 = res 409). '
                         'Applies to both the displayed labels and --start_residue/--end_residue.')
    ap.add_argument('--auto_offset',      action='store_true',
                    help='Automatically determine --residue_offset from the PDB id (FASTA header) '
                         'via the PDBe SIFTS mapping, using the deposited (author) numbering. '
                         'Requires network access (ebi.ac.uk). An explicit --residue_offset (!=0) takes priority. '
                         'If the PDB cannot be mapped, offset 0 is used with a warning.')

    # Layout
    ap.add_argument('--colormap', default='RdPu',
                     help="Matplotlib colormap name used for the probability heatmap (default: RdPu).")
    ap.add_argument('--figsize', nargs=2, type=float, default=[18, 10], metavar=('W', 'H'),
                     help="Figure size in inches as two numbers: width height (default: 18 10).")
    ap.add_argument('--cpu', action='store_true',
                     help="Force inference on CPU even if a CUDA GPU is available.")

    args   = ap.parse_args()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load model --
    model = load_model(args.model, device)

    # -- 2. Load dataset --
    data, protein_indices = load_dataset(
        args.data, args.protein_metadata, args.protein_name
    )
    if len(protein_indices) == 0:
        print(f"[ERROR] No window found for '{args.protein_name}'.")
        sys.exit(1)

    # -- 3. Find matching windows --
    matched_indices, _ = find_matching_windows(
        data, protein_indices, args.window_seq
    )
    if len(matched_indices) == 0:
        print(f"[ERROR] No window matches '{args.window_seq}'.")
        sys.exit(1)

    # -- 4. FASTA --
    fasta_seq = None
    if args.fasta and Path(args.fasta).exists():
        fasta_seq = read_fasta(args.fasta)
        print(f"\nFASTA: {len(fasta_seq)} residues")
    elif args.sequence_scan:
        print("[ERROR] --sequence_scan requires --fasta. Sequence scan disabled.")
        args.sequence_scan = False

    # -- 4a. Automatic numbering offset (optional) --
    #   canonical_residue = 1-based_FASTA_position + residue_offset
    #   an explicit --residue_offset (!=0) wins; otherwise try SIFTS if --auto_offset.
    if args.auto_offset and args.residue_offset == 0:
        pdb_for_offset = _extract_pdb_id_from_fasta(args.fasta) if args.fasta else None
        if pdb_for_offset is None:
            try:
                import json as _json
                with open(args.protein_metadata) as _f:
                    _md = _json.load(_f)
                for _v in _md.values():
                    cand = _v.get('pdb_id') if isinstance(_v, dict) else _v
                    if cand:
                        pdb_for_offset = str(cand)[:4]
                        break
            except Exception:
                pass
        off = auto_residue_offset(pdb_for_offset)
        if off is not None:
            args.residue_offset = off
            print(f"  [auto_offset] {pdb_for_offset}: residue_offset = {off}  "
                  f"(canonical_residue = FASTA_pos + {off})")
        else:
            print(f"  [auto_offset] could not be determined for {pdb_for_offset}; "
                  f"using residue_offset = {args.residue_offset}.")
    elif args.auto_offset and args.residue_offset != 0:
        print(f"  [auto_offset] ignored: --residue_offset={args.residue_offset} set manually.")

    # -- 4b. Filter windows by start_residue --
    if fasta_seq and args.start_residue > 1:
        win_size = data['aa_ids'].shape[1]
        center   = win_size // 2
        min_pos_0based = args.start_residue - 1 - args.residue_offset

        filtered_matched = []
        for idx in matched_indices:
            win_seq = window_to_seq(data['aa_ids'][idx])
            fasta_pos = fasta_seq.find(win_seq)
            if fasta_pos == -1:
                filtered_matched.append(idx)
                continue
            if fasta_pos + center >= min_pos_0based:
                filtered_matched.append(idx)

        n_removed = len(matched_indices) - len(filtered_matched)
        if n_removed > 0:
            print(f"\n  [start_residue={args.start_residue}] Excluded {n_removed} windows "
                  f"with central residue before residue {args.start_residue}.")
        matched_indices = filtered_matched
        print(f"  Windows kept (>= residue {args.start_residue}): {len(matched_indices)}")

        if len(matched_indices) == 0:
            print(f"[ERROR] No window left after start_residue={args.start_residue} filter.")
            sys.exit(1)

    # -- 4c. Filter windows by end_residue --
    if fasta_seq and args.end_residue is not None:
        win_size = data['aa_ids'].shape[1]
        center   = win_size // 2
        max_pos_0based = args.end_residue - 1 - args.residue_offset

        filtered_matched = []
        for idx in matched_indices:
            win_seq = window_to_seq(data['aa_ids'][idx])
            fasta_pos = fasta_seq.find(win_seq)
            if fasta_pos == -1:
                filtered_matched.append(idx)
                continue
            if fasta_pos + center <= max_pos_0based:
                filtered_matched.append(idx)

        n_removed = len(matched_indices) - len(filtered_matched)
        if n_removed > 0:
            print(f"\n  [end_residue={args.end_residue}] Excluded {n_removed} windows "
                  f"with central residue after residue {args.end_residue}.")
        matched_indices = filtered_matched
        print(f"  Windows kept (<= residue {args.end_residue}): {len(matched_indices)}")

        if len(matched_indices) == 0:
            print(f"[ERROR] No window left after end_residue={args.end_residue} filter.")
            sys.exit(1)

    # -- 5. Probability matrix --
    prob_matrix = compute_probability_matrix(
        model, data, matched_indices, device,
        mask_token_aa=args.mask_token_aa,
        mask_token_cl=args.mask_token_cl
    )

    # -- 6. Labels --
    with open(args.protein_metadata) as f:
        prot_meta_full = json.load(f)

    column_labels = build_column_labels(
        matched_indices, data, fasta_seq, args.window_seq,
        protein_metadata=prot_meta_full, residue_offset=args.residue_offset
    )

    # -- 7. Probability heatmap (original) --
    print_summary(prob_matrix, column_labels)
    plot_heatmap(
        prob_matrix, column_labels, args.window_seq,
        output_path=str(out_dir / 'heatmap_prob.png'),
        figsize=tuple(args.figsize), colormap=args.colormap,
        protein_name=args.protein_name
    )

    # -- 8. DeltaLogP landscape --
    delta_logp, wt_idx = compute_delta_logp(prob_matrix, args.window_seq)
    plot_delta_logp(
        delta_logp, column_labels, args.window_seq, wt_idx,
        output_path=str(out_dir / 'heatmap_delta_logp.png'),
        figsize=tuple(args.figsize),
        protein_name=args.protein_name
    )

    # -- 9. Per-window metrics --
    entropy    = compute_entropy_profile(prob_matrix)
    core_score = compute_core_score(delta_logp)
    tolerance  = compute_mutational_tolerance(delta_logp, threshold=-2.0)

    # Determine the breaker threshold: adaptive percentile (recommended) or fixed
    if args.breaker_threshold is not None:
        effective_threshold = args.breaker_threshold
        breakers, _  = find_breakers(delta_logp, threshold=effective_threshold, percentile=None)
    else:
        breakers, effective_threshold = find_breakers(
            delta_logp, percentile=args.breaker_percentile
        )

    # -- analysis_summary, embedding and bimodal threshold disabled --
    is_true_emb = False

    # -- 11. Window report --
    report = build_report(
        delta_logp, entropy, core_score, tolerance,
        column_labels, args.window_seq, wt_idx,
        breaker_threshold=effective_threshold,
        design_threshold=args.design_threshold
    )
    save_report(report, out_dir)

    # -- 12. Sequence scan (optional, paper figure) --
    if args.sequence_scan and fasta_seq:
        print(f"\n{'='*60}")
        print("SEQUENCE SCAN - Aggregation Mutability Landscape")
        print(f"  Scanning {len(fasta_seq)} residues of {args.protein_name}...")
        print(f"  Windows in dataset: {len(protein_indices)}")
        print('='*60)

        scan_results = sequence_scan(
            model, data, protein_indices, fasta_seq, device,
            mask_token_aa=args.mask_token_aa,
            mask_token_cl=args.mask_token_cl,
            start_residue=args.start_residue,
            end_residue=args.end_residue,
            residue_offset=args.residue_offset
        )

        if scan_results is not None:
            plot_aggregation_mutability_landscape(
                scan_results,
                protein_name=args.protein_name,
                output_path=str(out_dir / 'aggregation_mutability_landscape.png'),
                figsize=(max(22, len(scan_results['positions']) * 0.35 + 6), 12)
            )
            save_scan_report(scan_results, args.protein_name, out_dir)

    # -- Final summary --
    print(f"\n{'='*60}")
    print(f"[OK] Full analysis saved to: {out_dir}")
    print(f"  heatmap_prob.png          <- probability (original)")
    print(f"  heatmap_delta_logp.png    <- DeltaLogP landscape [main fig.]")
    print(f"  analysis_report.json      <- numeric summary")
    print(f"  analysis_report.csv       <- per-window table")
    if args.sequence_scan:
        print(f"  aggregation_mutability_landscape.png  <- PAPER FIGURE (3 panels)")
        print(f"  hotspot_index_distribution.png <- standalone bimodal distribution")
        print(f"  sequence_scan_report.csv              <- tabular landscape")
    print(f"\n  Breaker threshold used: {effective_threshold:.3f} "
          f"({'adaptive p' + str(args.breaker_percentile) if args.breaker_threshold is None else 'fixed'})")
    if args.start_residue > 1:
        print(f"  start_residue filter: >= {args.start_residue}")
    if args.end_residue is not None:
        print(f"  end_residue filter:   <= {args.end_residue}")
    print(f"{'='*60}")
    print("[OK] Done!")


if __name__ == '__main__':
    main()
