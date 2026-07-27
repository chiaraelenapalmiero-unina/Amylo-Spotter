"""
Heatmap probabilità di mutazione + Analisi mutazionale completa
================================================================

Estende mutation_heatmap1_or.py con:

  1. Heatmap probabilità (originale, invariata)
  2. ΔlogP mutational landscape  (20×N matrix, per ogni finestra)
  3. Entropy profile              (rigidità vs flessibilità per posizione)
  4. Breaker mutations            (ΔlogP < threshold, default -5)
  5. Designable positions         (mutazioni tollerate, |ΔlogP| < 1)
  6. Hotspot detection            (entropy > 75° percentile)
  7. Fibril core score            (mean |ΔlogP|, posizioni conservate)
  8. Embedding UMAP/PCA           (clustering strutturale delle finestre)
  9. Summary report               (JSON + CSV)

Usage:
    python mutation_heatmap_extended.py \\
        --model    teachers/teacher_α-synuclein/best_teacher.pt \\
        --data     windows_all_local.npz \\
        --protein_metadata proteins_metadata_with_class.json \\
        --rotamer_csv rotamer_localmax_maxdens.csv \\
        --fasta    fasta1.fasta \\
        --window_seq VGGAV \\
        --output   results/asynuclein_VGGAV
        python mutation_heatmap_extended_old.py        --model teachers/teacher_/best_teacher.pt        --protein_name "α-synuclein"       --data     windows_all_local.npz         --protein_metadata proteins_metadata_with_class.json         --rotamer_csv rotamer_localmax_maxdens.csv         --fasta  ./fasta_file/aS_7v49.fasta         --window_seq VGGAV         --output   resultsalpha_syn_VVGAV --sequence_scan --start_residue 38  --end_residue 95
        
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


    Tutti gli output vengono salvati nella cartella --output:
        heatmap_prob.png          ← heatmap probabilità (originale)
        heatmap_delta_logp.png    ← landscape ΔlogP
        entropy_profile.png       ← entropy per finestra
        core_score.png            ← fibril core score
        breaker_map.png           ← mappa breaker mutations
        embedding_umap.png        ← clustering strutturale (se sklearn/umap)
        analysis_report.json      ← summary numerico completo
        analysis_report.csv       ← tabella per posizione

        set -euo pipefail
 
MODEL=teacher_pathological_vs_functional/best_teacher.pt
DATA=windows_all_local.npz
META=proteins_metadata_with_class.json
ROTAMER=rotamer_localmax_maxdens.csv
 
# formato: nome_proteina|fasta|window_seq_iniziale
declare -a PROTEINS=(
  "Antimicrobial peptide|./fasta_file/Antimicrobial peptide_7QV5.fasta|DLIRK"
  "CPEB3|./fasta_file/CPEB3_8SPA.fasta|LSPSF"
  "CsgA|./fasta_file/CsgA_8ENQ.fasta|PNSEL"
  "HET-s (218-289)|./fasta_file/HET-s_2RNM.fasta|MKIDA"
  "Mrip3|./fasta_file/Mrip3_6JPD.fasta|VFNNC"
  "Nup98|./fasta_file/Nup98_7Q64.fasta|TGTAN"
  "RIPK1|./fasta_file/RIPK1_8IB0.fasta|TIYNS"
  "β-endorphin|./fasta_file/β-endorphin_6TUB.fasta|YGGFM"
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
  echo "=== $NAME — $LABEL ($WIN) ==="
  python "$SCRIPT" \
    --model "$MODEL" \
    --data "$DATA" \
    --protein_metadata "$META" \
    --rotamer_csv "$ROTAMER" \
    --fasta "$FASTA" \
    --protein_name "$NAME — $LABEL" \
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
  echo "=== $NAME — $LABEL ($WIN) ==="
  python "$SCRIPT" \
    --model "$MODEL" \
    --data "$DATA" \
    --protein_metadata "$META" \
    --rotamer_csv "$ROTAMER" \
    --fasta "$FASTA" \
    --protein_name "$NAME — $LABEL" \
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

# topk_model2.py vive in ../model (usato più sotto in load_model())
sys.path.append(str(Path(__file__).resolve().parent.parent / "model"))


# ── Helper: salvataggio multi-formato ────────────────────────────────────────
def _save_multiformat(fig, output_path: str, label: str = ""):
    """
    Salva la figura in PNG (600 dpi), SVG, PDF e TIFF (600 dpi, LZW)
    dentro una sottocartella 'figures/' nella stessa directory di output_path.
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
    print(f"✓{tag} Salvata: {dest}  (.png / .svg / .pdf / .tiff)")

# ── Costanti AA ─────────────────────────────────────────────────────────────
AA_ORDER   = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX  = {aa: i for i, aa in enumerate(AA_ORDER)}
IDX_TO_AA  = {i: aa for i, aa in enumerate(AA_ORDER)}
AA_NAMES   = {i: aa for i, aa in enumerate(AA_ORDER)}
AA_TO_NUM  = {v: k for k, v in AA_NAMES.items()}

# Proprietà biofisiche per annotazioni (stesso ordine AA_ORDER)
AA_PROPERTIES = {
    # Idrofobicità (Kyte-Doolittle, normalizzata 0-1)
    'hydrophobic': ['A','C','F','I','L','M','V','W'],
    # Polari/carichi
    'charged_pos': ['K','R','H'],
    'charged_neg': ['D','E'],
    'polar':       ['N','Q','S','T','Y'],
    # Speciali
    'proline':     ['P'],
    'glycine':     ['G'],
}

# Colori per proprietà (usati nelle annotazioni)
AA_PROP_COLOR = {}
for aa in AA_ORDER:
    if aa in AA_PROPERTIES['hydrophobic']:  AA_PROP_COLOR[aa] = '#1A3A64'
    elif aa in AA_PROPERTIES['charged_pos']: AA_PROP_COLOR[aa] = '#8B0000'
    elif aa in AA_PROPERTIES['charged_neg']: AA_PROP_COLOR[aa] = '#006400'
    elif aa in AA_PROPERTIES['polar']:       AA_PROP_COLOR[aa] = '#4B0082'
    elif aa in AA_PROPERTIES['proline']:     AA_PROP_COLOR[aa] = '#8B4513'
    elif aa in AA_PROPERTIES['glycine']:     AA_PROP_COLOR[aa] = '#2F4F4F'
    else:                                    AA_PROP_COLOR[aa] = '#333333'


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCCO 1 — Funzioni originali (invariate al 100%)
# ═══════════════════════════════════════════════════════════════════════════════

def read_fasta(fasta_path):
    seq = ''
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    return seq


def _extract_pdb_id_from_fasta(fasta_path):
    """Estrae il PDB id dall'header RCSB tipo '>2RNM_1|Chains...' -> '2RNM'."""
    try:
        with open(fasta_path) as f:
            for line in f:
                if line.startswith('>'):
                    token = line[1:].strip().split('|')[0]   # es. '2RNM_1'
                    pdb = token.split('_')[0].strip()
                    return pdb.upper() if pdb else None
    except Exception:
        pass
    return None


def _parse_sifts_offset(data, pdb_key):
    """
    Dalla risposta SIFTS PDBe calcola offset = numero_autore - posizione_SEQRES(=FASTA).
    Usa il segmento UniProt più lungo. Ritorna int o None.
    """
    unp_block = data.get(pdb_key, {}).get('UniProt', {})
    best_len, offset = -1, None
    for _acc, acc_data in unp_block.items():
        for seg in acc_data.get('mappings', []):
            start    = seg.get('start', {})
            res_num  = start.get('residue_number')          # 1-based nel SEQRES = pos FASTA
            auth_num = start.get('author_residue_number')   # numerazione depositata (canonica)
            if res_num is None or auth_num is None:
                continue
            seg_len = seg.get('unp_end', 0) - seg.get('unp_start', 0)
            if seg_len > best_len:
                best_len = seg_len
                offset   = int(auth_num) - int(res_num)
    return offset


def auto_residue_offset(pdb_id, timeout=20, _cache={}):
    """
    Determina in automatico l'offset (residuo_canonico - posizione_FASTA_1based)
    dalla mappatura SIFTS di PDBe, usando il NUMERO D'AUTORE del residuo, cioè la
    numerazione depositata nel PDB (HET-s 218-289, RIPK3 409-486, beta-endorfina 1-31).
    Richiede connessione a ebi.ac.uk. Ritorna int oppure None se non determinabile.
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
        print(f"  ⚠️  auto-offset: SIFTS non raggiungibile per {key} ({e}).")
        offset = None
    _cache[key] = offset
    return offset


def load_model(model_path, device):
    from topk_model2 import ProteinMLM_v2

    print(f"\nCaricamento checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_config' in ckpt:
        cfg       = ckpt['model_config']
        state_key = 'model_state_dict'
    else:
        raise KeyError("Il checkpoint non contiene 'model_config'.")

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
    print("✓ Modello caricato")
    return model


def _normalize_pdb_id(pdb_id: str) -> str:
    """
    Riduce un pdb_id al codice PDB canonico a 4 caratteri (case-insensitive).
    Esempi:
        '8BFA_filtered_new_r5c23d4.836a0.648'     → '8bfa'
        '8bfa_filtered_new_r5c23d4.836a0.648_new' → '8bfa'
        '2MPZ_filtered_new_r5c23d4.8a0.0'         → '2mpz'
    """
    return pdb_id.split('_')[0].lower()


def load_dataset(npz_path, protein_metadata_path, target_protein):
    print(f"\nCaricamento dataset: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    print(f"  Chiavi NPZ: {list(data.keys())}")

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
    print(f"  Finestre totali nel dataset: {data['aa_ids'].shape[0]}")
    print(f"  Finestre di '{target_protein}' (pre-dedup): {len(all_indices)}")

    # ── Deduplicazione PDB ────────────────────────────────────────────────
    # Alcuni PDB compaiono due volte con naming diverso (es. '8BFA_filtered...'
    # e '8bfa_filtered..._new'). Per ogni codice PDB canonico (4 char, lowercase)
    # manteniamo TUTTE le finestre della prima versione incontrata e scartiamo
    # TUTTE le finestre delle versioni duplicate.
    #
    # Strategia:
    #   1. Raggruppa gli indici per pdb_id completo  →  {pdb_id: [idx, ...]}
    #   2. Per ogni codice 4-char unico, scegli la versione con l'indice più basso
    #      (convenzione riproducibile e deterministica).
    #   3. Includi solo gli indici di quella versione.

    # Passo 1: raggruppa per pdb_id completo
    pdb_id_to_indices: dict[str, list[int]] = {}
    for idx in all_indices:
        raw_pdb = get_pdb_id(protein_metadata.get(str(idx), {}))
        pdb_id_to_indices.setdefault(raw_pdb, []).append(idx)

    # Passo 2: per ogni codice canonico, scegli il pdb_id con l'indice minimo
    code_to_chosen_pdb: dict[str, str] = {}   # codice_4char → pdb_id scelto
    for raw_pdb in pdb_id_to_indices:
        code = _normalize_pdb_id(raw_pdb) if raw_pdb else f'__unknown__'
        min_idx = min(pdb_id_to_indices[raw_pdb])
        if code not in code_to_chosen_pdb:
            code_to_chosen_pdb[code] = raw_pdb
        else:
            current_min = min(pdb_id_to_indices[code_to_chosen_pdb[code]])
            if min_idx < current_min:
                code_to_chosen_pdb[code] = raw_pdb

    # Passo 3: raccogli gli indici delle versioni scelte
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
        print(f"  ⚠️  Deduplicazione PDB: scartate {len(skipped_versions)} versioni duplicate:")
        for dup_pdb, n_win, kept_pdb in skipped_versions:
            print(f"       RIMOSSA  ({n_win:3d} finestre): '{dup_pdb}'")
            print(f"       MANTENUTA              : '{kept_pdb}'")
    else:
        print(f"  ✓ Nessun PDB duplicato trovato.")

    print(f"  Finestre di '{target_protein}' (post-dedup): {len(dedup_indices)}  "
          f"({len(code_to_chosen_pdb)} strutture PDB uniche)")

    return data, dedup_indices


def window_to_seq(aa_array):
    return ''.join(AA_NAMES.get(int(a), '?') for a in aa_array)


def find_matching_windows(data, asynuclein_indices, window_seq):
    matched_indices = []
    positions_info  = []
    aa_ids          = data['aa_ids']

    for idx in asynuclein_indices:
        seq = window_to_seq(aa_ids[idx])
        if seq == window_seq:
            matched_indices.append(idx)
            positions_info.append({'global_idx': idx, 'sequence': seq})

    print(f"\n  Finestre che matchano '{window_seq}': {len(matched_indices)}")
    return matched_indices, positions_info


@torch.no_grad()
def predict_probs_single(model, data, global_idx, device, mask_token_aa, mask_token_cl):
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
    all_probs = []
    print(f"\nCalcolo probabilità per {len(matched_indices)} finestre...")

    for i, idx in enumerate(matched_indices):
        probs = predict_probs_single(model, data, idx, device,
                                     mask_token_aa, mask_token_cl)
        all_probs.append(probs)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(matched_indices)}] idx={idx} ✓")

    matrix = np.column_stack(all_probs)   # (20, N)
    print(f"  Matrice finale: {matrix.shape[0]} AA × {matrix.shape[1]} finestre")
    return matrix


def build_column_labels(matched_indices, data, fasta_seq, window_seq,
                        protein_metadata=None, residue_offset=0):
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
    """Heatmap originale — invariata."""
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
        f'Mutation Probability — {protein_name}\nWindow: {window_seq}  '
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
    mean_probs = matrix.mean(axis=1)
    top5_idx   = np.argsort(mean_probs)[::-1][:5]
    print("\n── Riepilogo probabilità medie (residuo centrale mascherato) ──")
    print(f"{'AA':<6} {'Prob media':>12}")
    print("-" * 20)
    for idx in top5_idx:
        print(f"  {AA_NAMES[idx]:<4}  {mean_probs[idx]:.4f}")
    print(f"\nStatistiche matrice:")
    print(f"  Min:  {matrix.min():.4f}  Max: {matrix.max():.4f}  "
          f"Mean: {matrix.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCCO 2 — Nuove analisi
# ═══════════════════════════════════════════════════════════════════════════════

def compute_delta_logp(prob_matrix, wt_sequence):
    """
    Calcola la matrice ΔlogP per ogni finestra.

    ΔlogP(i, a) = log P(a | context_i) - log P(WT_i | context_i)

    Args:
        prob_matrix  : np.array (20, N_windows)
        wt_sequence  : stringa della finestra, es. "VGGAV"
                       Usiamo il residuo CENTRALE come WT per ogni finestra
                       (tutte le finestre hanno la stessa sequenza centrale)
    
    Returns:
        delta_logp  : np.array (20, N_windows)
        wt_idx      : int — indice dell'AA centrale WT
    """
    center     = len(wt_sequence) // 2
    wt_aa      = wt_sequence[center]
    wt_idx     = AA_TO_IDX[wt_aa]

    # log P(mut) - log P(WT) per ogni colonna (finestra)
    log_probs  = np.log(prob_matrix + 1e-9)          # (20, N)
    wt_log     = log_probs[wt_idx, :]                # (N,)
    delta_logp = log_probs - wt_log[np.newaxis, :]   # (20, N)

    return delta_logp, wt_idx


def compute_entropy_profile(prob_matrix):
    """
    H(i) = -Σ_a P(a|i) log P(a|i)
    
    Returns:
        entropy : np.array (N_windows,) — una entropia per finestra
    """
    entropy = -(prob_matrix * np.log(prob_matrix + 1e-9)).sum(axis=0)  # (N,)
    return entropy


def find_breakers(delta_logp, threshold=-5.0, percentile=None):
    """
    Residui con ΔlogP < threshold → sono breaker della fibrilla.

    Se percentile è fornito (es. 5), la soglia viene calcolata come il
    percentile più basso della distribuzione ΔlogP, ignorando la diagonale
    WT (ΔlogP = 0 per definizione). Questo è robusto rispetto alla scala
    del modello e produce sempre il 5% peggiore delle mutazioni.

    Args:
        delta_logp : np.array (20, N_windows)
        threshold  : soglia fissa (usata se percentile è None)
        percentile : se fornito, sovrascrive threshold con il percentile-esimo
                     valore della distribuzione (default None → usa threshold)

    Returns:
        breakers          : list of (window_idx, aa_letter, delta_logp_value)
                            sorted per delta_logp crescente (peggiori prima)
        effective_threshold : float — soglia effettivamente usata
    """
    if percentile is not None:
        # Esclude esattamente gli zeri della diagonale WT (ΔlogP WT = 0 per definizione).
        # NON esclude mutazioni neutrali vicine a zero: quelle sono informative.
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
    Finestre dove ci sono ≥ min_tolerated mutazioni con ΔlogP > threshold.
    Posizioni di design → puoi modificare senza destabilizzare la fibrilla.
    
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
    Core score = media di |ΔlogP| su tutte le mutazioni per ogni finestra.
    Alto core score → posizione molto conservata (fibril core).
    
    Returns:
        core_score : np.array (N_windows,)
    """
    return np.abs(delta_logp).mean(axis=0)  # (N,)


def compute_mutational_tolerance(delta_logp, threshold=-2.0):
    """
    Numero di mutazioni con ΔlogP > threshold per ogni finestra.
    
    Returns:
        tolerance : np.array (N_windows,) di int
    """
    return (delta_logp > threshold).sum(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCCO 3 — Embedding extraction + UMAP/PCA
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embedding(model, data, global_idx, device, mask_token_aa, mask_token_cl):
    """
    Estrae l'embedding strutturale VERO dal modello: il vettore h prodotto
    dall'attenzione BiLSTM, PRIMA della testa di classificazione.

    Strategia a 3 livelli (dal più informativo al fallback):

    Livello 1 — return_embedding=True (TeacherModel/FibrilBreakerModel):
        output['h'] → vettore (d_hidden*2,) — il vero embedding strutturale.
        Questo è il più ricco: include contesto LSTM + attenzione.

    Livello 2 — hook PyTorch sul layer 'dropout' pre-head:
        Aggancia un forward hook sull'ultimo dropout prima di head_aa.
        Cattura h_dropped (d_hidden*2,) senza modificare il modello.
        Funziona su ProteinMLM_v2 standard senza return_embedding.

    Livello 3 — fallback logit-space:
        Se tutto fallisce, usa softmax(logits) come proxy.
        Questo NON è un vero embedding strutturale (è solo la distribuzione AA),
        e viene segnalato con un warning.

    Returns:
        h_vec        : np.array (D,)
        is_true_emb  : bool — True se siamo al livello 1 o 2
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

    # ── Livello 1: return_embedding API (Teacher / FibrilBreaker) ──────────
    try:
        output = model(batch, return_embedding=True)
        if isinstance(output, dict) and 'h' in output and output['h'] is not None:
            h = output['h'][0].cpu().numpy()
            return h, True
    except Exception:
        pass

    # ── Livello 2: forward hook sull'input di head_aa (pre-head) ──────────
    # Strategia robusta: invece di assumere model.dropout esista (non garantito),
    # aggancia il hook direttamente sull'INGRESSO del primo Linear di head_aa.
    # head_aa è sempre presente in ProteinMLM_v2, quindi questo è garantito.
    # inp[0] all'ingresso del primo Linear è esattamente h_dropped (d_hidden*2,).
    captured = {}

    def _hook(module, inp, out):
        captured['h'] = inp[0].detach()

    hook_handle = None
    try:
        # Trova il primo layer Linear dentro head_aa — sempre presente in ProteinMLM_v2
        # In alternativa, se esiste model.dropout lo usiamo (più pulito),
        # altrimenti usiamo head_aa[0] (il primo Linear).
        target_module = None

        # Prima scelta: self.dropout esplicito (ProteinMLM_v2 standard)
        if hasattr(model, 'dropout') and isinstance(model.dropout, nn.Dropout):
            target_module = model.dropout
            hook_on_input = True   # vogliamo inp[0] = h prima del dropout

        # Seconda scelta: primo Linear di head_aa (sempre presente)
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

    # ── Livello 3: fallback logit-space (NON è un vero embedding) ──────────
    print("  ⚠️  FALLBACK: usando distribuzione AA come proxy embedding "
          "(non è un vero embedding strutturale — il clustering sarà meno informativo).")
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
    Raccoglie embedding per tutte le finestre.

    Returns:
        embeddings    : np.array (N, D)
        is_true_emb   : bool — True se tutti gli embedding sono strutturali veri
    """
    embeddings    = []
    true_emb_flags = []
    print(f"\nEstrazione embeddings per {len(matched_indices)} finestre...")

    for i, idx in enumerate(matched_indices):
        h, is_true = extract_embedding(model, data, idx, device,
                                       mask_token_aa, mask_token_cl)
        embeddings.append(h)
        true_emb_flags.append(is_true)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(matched_indices)}] ✓")

    is_true_emb = all(true_emb_flags)
    quality     = "strutturale (LSTM pre-head)" if is_true_emb else "proxy logit-space"
    print(f"  Embedding quality: {quality}")
    return np.array(embeddings), is_true_emb


def reduce_embeddings(embeddings, method='pca'):
    """
    Riduce embedding a 2D con PCA (sempre disponibile) o UMAP (opzionale).
    
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
            print("  ⚠️  umap-learn non installato, uso PCA.")

    from sklearn.decomposition import PCA
    pca       = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(embeddings)
    return coords_2d, f'PCA (var. spiegata: {pca.explained_variance_ratio_.sum():.1%})'


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCCO 6 — Sequence scan: aggregation mutability landscape
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_probs_from_window_data(model, data, global_idx, device,
                                   mask_token_aa, mask_token_cl):
    """
    Identica a predict_probs_single — restituisce probs (20,) per una finestra.
    Resa separata per chiarezza nella sequence_scan pipeline.
    """
    return predict_probs_single(model, data, global_idx, device,
                                mask_token_aa, mask_token_cl)


def sequence_scan(model, data, all_protein_indices, fasta_seq, device,
                  mask_token_aa=20, mask_token_cl=0, start_residue=1, end_residue=None,
                  residue_offset=0):
    """
    Scansiona TUTTE le posizioni della proteina mascherando ogni finestra.

    Per ogni posizione nella sequenza FASTA:
      - trova le finestre del dataset che hanno il residuo centrale in quella posizione
      - calcola la probabilità media su tutte le strutture disponibili
      - calcola ΔlogP, entropy, core_score per quella posizione

    Questo produce l'aggregation mutability landscape lungo tutta la sequenza:
    non più per un singolo frammento (VGGAV), ma per ogni residuo della proteina.

    Args:
        model               : ProteinMLM_v2 loaded
        data                : NPZ dataset dict
        all_protein_indices : lista di tutti gli indici del dataset per questa proteina
        fasta_seq           : stringa sequenza completa (es. α-syn, 140 aa)
        device              : torch.device

    Returns:
        scan_results : dict con chiavi:
            'positions'    : list[int] — posizioni 1-based con almeno 1 finestra
            'wt_aas'       : list[str] — AA WT per ogni posizione
            'mean_probs'   : np.array (L_covered, 20) — prob media
            'delta_logp'   : np.array (L_covered, 20) — ΔlogP per posizione
            'entropy'      : np.array (L_covered,)
            'core_score'   : np.array (L_covered,)
            'tolerance'    : np.array (L_covered,) — n mutazioni tollerate
            'n_structures' : list[int] — n strutture per posizione
    """
    aa_ids   = data['aa_ids']
    win_size = aa_ids.shape[1]
    center   = win_size // 2

    # Mappa: posizione FASTA 0-based → lista di global_idx nel dataset
    # FIX: trova TUTTE le occorrenze della finestra nella FASTA, non solo la prima.
    # α-syn contiene ripetizioni di sequenza → una finestra può mappare su più posizioni.
    pos_to_indices = defaultdict(list)

    for idx in all_protein_indices:
        win_seq = window_to_seq(aa_ids[idx])
        # Itera su tutte le occorrenze nella FASTA
        start = 0
        while True:
            fasta_pos = fasta_seq.find(win_seq, start)
            if fasta_pos == -1:
                break
            center_pos = fasta_pos + center  # posizione 0-based del residuo centrale
            pos_to_indices[center_pos].append(idx)
            start = fasta_pos + 1  # avanza di 1 per trovare occorrenze sovrapposte

    if not pos_to_indices:
        print("  ⚠️  Sequence scan: nessuna finestra mappata sulla FASTA.")
        return None

    # Filtra posizioni prima di start_residue (in numerazione canonica)
    if start_residue > 1:
        min_pos_0based = start_residue - 1 - residue_offset
        filtered_keys = [p for p in pos_to_indices if p >= min_pos_0based]
        n_removed = len(pos_to_indices) - len(filtered_keys)
        if n_removed > 0:
            print(f"  [start_residue={start_residue}] Escluse {n_removed} posizioni "
                  f"prima del residuo {start_residue}.")
        pos_to_indices = {k: pos_to_indices[k] for k in filtered_keys}

    # Filtra posizioni dopo end_residue (in numerazione canonica)
    if end_residue is not None:
        max_pos_0based = end_residue - 1 - residue_offset
        filtered_keys = [p for p in pos_to_indices if p <= max_pos_0based]
        n_removed = len(pos_to_indices) - len(filtered_keys)
        if n_removed > 0:
            print(f"  [end_residue={end_residue}] Escluse {n_removed} posizioni "
                  f"dopo il residuo {end_residue}.")
        pos_to_indices = {k: pos_to_indices[k] for k in filtered_keys}

    sorted_positions = sorted(pos_to_indices.keys())
    print(f"\n[SEQUENCE SCAN] Posizioni coperte: {len(sorted_positions)} / {len(fasta_seq)}")
    print(f"  Finestre totali da processare: "
          f"{sum(len(v) for v in pos_to_indices.values())}")

    # ── Pre-calcola probs per TUTTI gli indici unici in un unico passaggio ──
    # Invece di chiamare predict_probs per ogni (posizione, struttura) separatamente,
    # calcoliamo una volta sola tutte le probabilità per tutti gli indici unici.
    # Questo riduce il numero di forward pass da O(N_windows) a O(N_unique_indices).
    all_unique_indices = sorted(set(
        idx for indices in pos_to_indices.values() for idx in indices
    ))
    print(f"  Indici unici (forward passes): {len(all_unique_indices)}")

    idx_to_probs = {}
    for i, idx in enumerate(all_unique_indices):
        p = predict_probs_from_window_data(model, data, idx, device,
                                           mask_token_aa, mask_token_cl)
        idx_to_probs[idx] = p
        if (i + 1) % 50 == 0 or i == len(all_unique_indices) - 1:
            print(f"  [{i+1}/{len(all_unique_indices)}] forward passes ✓")

    positions    = []
    wt_aas       = []
    mean_probs   = []
    n_structures = []

    for pos_0 in sorted_positions:
        indices = pos_to_indices[pos_0]
        wt_aa   = fasta_seq[pos_0]

        if wt_aa not in AA_TO_IDX:
            continue  # salta residui non standard

        # Media probabilità: usa cache, nessun forward pass extra
        probs_list = [idx_to_probs[idx] for idx in indices if idx in idx_to_probs]
        if not probs_list:
            continue

        mean_p = np.mean(probs_list, axis=0)  # (20,)

        positions.append(pos_0 + 1 + residue_offset)  # numerazione canonica
        wt_aas.append(wt_aa)
        mean_probs.append(mean_p)
        n_structures.append(len(probs_list))

    mean_probs_arr = np.array(mean_probs)   # (L, 20)

    # ΔlogP per ogni posizione (WT specifico di quella posizione)
    log_probs = np.log(mean_probs_arr + 1e-9)   # (L, 20)
    delta_logp_scan = np.zeros_like(log_probs)
    for i, wt_aa in enumerate(wt_aas):
        wt_idx_i = AA_TO_IDX[wt_aa]
        delta_logp_scan[i] = log_probs[i] - log_probs[i, wt_idx_i]

    # Metriche
    entropy_scan    = -(mean_probs_arr * np.log(mean_probs_arr + 1e-9)).sum(axis=1)
    core_score_scan = np.abs(delta_logp_scan).mean(axis=1)
    tolerance_scan  = (delta_logp_scan > -2.0).sum(axis=1)

    # ── Media CON segno (non valore assoluto) ────────────────────────────
    # Distingue due fenomeni che core_score (valore assoluto) confonde:
    #   mean_signed < 0  → il WT è (quasi) ottimale, le mutazioni sono
    #                      penalizzate → vero hotspot/breaker conservato
    #   mean_signed >= 0 → il WT stesso appare sub-ottimale secondo il
    #                      modello, quasi ogni sostituzione lo migliora →
    #                      posizione "tollerante", non un hotspot di
    #                      aggregazione nel senso classico
    mean_signed_scan = delta_logp_scan.mean(axis=1)

    print(f"  Posizioni con dati: {len(positions)}")
    print(f"  Strutture per posizione: min={min(n_structures)}, "
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
    Trova la soglia per la Hotspot Index come il punto in cui la
    densità della distribuzione scende per la prima volta vicino a zero
    dopo il picco dominante.

    Motivazione biologica: la distribuzione ha un unico grande picco
    (posizioni flessibili/tolleranti) seguito da una coda che si azzera
    intorno a un certo valore. Le posizioni a destra di quel punto sono
    le posizioni rigide/sensibili (outlier della coda).

    Algoritmo:
      1. Stima la densità con istogramma + smoothing gaussiano
      2. Trova il picco dominante (densità massima)
      3. Dopo il picco dominante, scorre a destra fino al primo bin in cui
         smoothed_density < zero_fraction * peak_density
         (default: 5% della densità del picco = "quasi zero")
      4. Quel bin è il threshold

    Args:
        values            : array 1D dei valori di hotspot index
        n_bins            : numero di bin dell'istogramma (default 50)
        smooth_sigma      : sigma del filtro gaussiano (default 1.5)
        zero_fraction     : frazione della densità del picco considerata "zero"
                            (default 0.05 = 5%)
        fallback_percentile: percentile usato se l'algoritmo non converge

    Returns:
        threshold : float — soglia ottimale
        info      : dict con chiavi:
                      'method'           : 'density_near_zero' o 'percentile_fallback'
                      'peaks'            : [x del picco dominante]
                      'valley'           : threshold trovato (float o None)
                      'zero_level'       : densità considerata "zero" (5% del picco)
                      'bin_centers'      : array x dell'istogramma
                      'smoothed_density' : array densità smoothed
    """
    values = np.asarray(values, dtype=float)

    # ── 1. Istogramma + smooth ─────────────────────────────────────────────
    counts, bin_edges = np.histogram(values, bins=n_bins, density=True)
    bin_centers       = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    smoothed          = gaussian_filter1d(counts, sigma=smooth_sigma)

    # ── 2. Picco dominante ────────────────────────────────────────────────
    peak_idxs, _ = find_peaks(smoothed)

    info_base = {
        'bin_centers':       bin_centers,
        'smoothed_density':  smoothed,
    }

    if len(peak_idxs) == 0:
        thr = float(np.percentile(values, fallback_percentile))
        print(f"  ⚠️  Nessun picco trovato. Fallback: p{fallback_percentile} = {thr:.4f}")
        return thr, {**info_base, 'method': f'percentile_fallback_p{fallback_percentile}',
                     'peaks': [], 'valley': None, 'zero_level': None}

    dominant_idx      = int(peak_idxs[np.argmax(smoothed[peak_idxs])])
    peak_density      = float(smoothed[dominant_idx])
    zero_level        = zero_fraction * peak_density

    # ── 3. Prima discesa sotto zero_level dopo il picco dominante ─────────
    region_after = smoothed[dominant_idx:]
    below_idxs   = np.where(region_after < zero_level)[0]

    if len(below_idxs) == 0:
        # La coda non scende mai sotto la soglia: fallback percentile
        thr = float(np.percentile(values, fallback_percentile))
        print(f"  ⚠️  La densità non scende mai sotto {zero_fraction*100:.0f}% del picco. "
              f"Fallback: p{fallback_percentile} = {thr:.4f}")
        return thr, {**info_base, 'method': f'percentile_fallback_p{fallback_percentile}',
                     'peaks': [float(bin_centers[dominant_idx])],
                     'valley': None, 'zero_level': zero_level}

    threshold_bin_idx = dominant_idx + int(below_idxs[0])
    threshold         = float(bin_centers[threshold_bin_idx])

    print(f"  Picco dominante  : {bin_centers[dominant_idx]:.4f}  "
          f"(densità {peak_density:.4f})")
    print(f"  Livello 'zero'   : {zero_level:.4f}  ({zero_fraction*100:.0f}% del picco)")
    print(f"  ✓ Threshold (prima discesa a zero): {threshold:.4f}")

    return threshold, {**info_base,
                       'method':     'density_near_zero',
                       'peaks':      [float(bin_centers[dominant_idx])],
                       'valley':     threshold,
                       'zero_level': zero_level}



def _draw_sensitivity_distribution(ax, core_score, threshold, bimodal_info):
    """
    Disegna la distribuzione dei valori di Hotspot Index su un asse
    già esistente (usato sia nella figura principale che nel plot standalone).

    Layout:
      - Istogramma con fill bicolore (< threshold = light purple, > threshold = dark purple)
      - Curva KDE smoothed
      - Linee verticali ai picchi (verde tratteggiato)
      - Linea verticale al threshold (arancione tratteggiata) + shading
      - Annotazione testuale del valore threshold
    """
    bin_centers = bimodal_info['bin_centers']
    smoothed    = bimodal_info['smoothed_density']
    n_bins      = len(bin_centers)
    is_bimodal  = bimodal_info.get('valley') is not None
    zero_level  = bimodal_info.get('zero_level')
    method      = bimodal_info.get('method', '')

    # Palette coerente con la heatmap ΔlogP sopra (RdBu, lato rosso):
    COLOR_RIGID    = '#67001F'   # rosso scurissimo RdBu (residuo 42 nella heatmap)
    COLOR_FLEXIBLE = '#FDDBC7'   # rosa pallido RdBu (Y/A53 nella heatmap)
    COLOR_CURVE    = '#4A0015'   # ancora più scuro per la curva KDE
    COLOR_THR      = '#111111'   # nero per la dashed line threshold

    # ── Istogramma bicolore ───────────────────────────────────────────────
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    for cx, dens in zip(bin_centers, smoothed):
        color = COLOR_RIGID if cx > threshold else COLOR_FLEXIBLE
        ax.bar(cx, dens, width=bin_width * 0.92, color=color, alpha=0.55,
               edgecolor='none', align='center')

    # ── Curva KDE smoothed ────────────────────────────────────────────────
    ax.plot(bin_centers, smoothed, color=COLOR_CURVE, lw=2.2,
            label='Smoothed density', zorder=4)

    # ── Linea orizzontale del livello "quasi-zero" ────────────────────────
    if zero_level is not None:
        ax.axhline(zero_level, color='#888', linestyle=':', lw=1.2, alpha=0.7,
                   label=f'threshold ({zero_level:.3f})')

    # ── Picco dominante ────────────────────────────────────────────────────
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

    # ── Threshold: linea nera + shading + annotazione ─────────────────────
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
    ax.set_xlabel('Hotspot Index (mean |ΔlogP|)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14)
    method_str = 'near-zero cutoff' if method == 'density_near_zero' else 'p75 fallback'
    ax.set_title(f'Hotspot Index distribution  —  threshold: {method_str}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.7)
    ax.tick_params(labelsize=14)


def plot_sensitivity_distribution_standalone(core_score, threshold, bimodal_info,
                                             protein_name='Protein', output_path=None):
    """
    Figura standalone della distribuzione dei valori di Hotspot Index
    con il threshold bimodale evidenziato.

    Salvata come 'hotspot_index_distribution.png' nella cartella output.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='white')
    _draw_sensitivity_distribution(ax, core_score, threshold, bimodal_info)
    fig.suptitle(
        f'{protein_name} — Hotspot Index distribution',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if output_path:
        _save_multiformat(fig, output_path, label="hotspot_distribution")
    plt.close(fig)


def plot_aggregation_mutability_landscape(scan_results, protein_name,
                                          output_path=None, figsize=(22, 16)):
    """
    Figura principale del paper: aggregation mutability landscape.

    3 pannelli sovrapposti:
      1. ΔlogP heatmap (20 × L_covered)      — landscape completo
      2. Hotspot Index bar chart     — mean |ΔlogP| per posizione
      3. Distribuzione Hotspot Index — istogramma + KDE + threshold bimodale
    """
    positions   = scan_results['positions']
    wt_aas      = scan_results['wt_aas']
    delta_logp  = scan_results['delta_logp']    # (L, 20)
    core_score  = scan_results['core_score']     # (L,)

    L = len(positions)
    x = np.arange(L)
    tick_positions = list(range(0, L, 3))

    # ── Calcola threshold bimodale una sola volta ─────────────────────────
    core_thr, bimodal_info = find_sensitivity_threshold(core_score)
    method_label = (
        f"threshold = {core_thr:.3f}"
        if bimodal_info['method'] == 'density_near_zero'
        else f"p{75} fallback = {core_thr:.3f}"
    )

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs  = gridspec.GridSpec(3, 1, hspace=0.50, height_ratios=[3, 1, 1.4], figure=fig)

    # ── Pannello 1: ΔlogP heatmap ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    delta_T = delta_logp.T   # (20, L) per imshow
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
        f'Mutability Landscape — {protein_name}\n'
        f'ΔlogP for each position in the sequence (black dot = WT)',
        fontsize=13, fontweight='bold', pad=12
    )

    cbar = plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.01)
    cbar.set_label('ΔlogP', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # ── Pannello 2: Hotspot Index bar chart ──────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # mean_signed distingue la direzione dello spostamento medio:
    #   < 0  → WT (quasi) ottimale, mutazioni penalizzate → vero hotspot/breaker
    #   >= 0 → WT sub-ottimale secondo il modello, mutazioni favorite → posizione
    #          "tollerante", core_score alto ma NON un hotspot di aggregazione
    mean_signed = scan_results.get('mean_signed', delta_logp.mean(axis=1))

    COLOR_BREAKER  = "#9B1111"   # rosso scuro — vero hotspot (WT ottimale, muta = peggio)
    COLOR_TOLERANT = "#1B5E6B"   # petrolio — WT sub-ottimale (muta = meglio)
    COLOR_NEUTRAL  = "#EEC2A9"   # pesca chiaro — posizione flessibile, non sensibile

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

    # ── Pannello 3: Distribuzione Hotspot Index ─────────────────
    # Non condivide l'asse X con i pannelli superiori (asse diverso: valori, non posizioni)
    ax3 = fig.add_subplot(gs[2])
    _draw_sensitivity_distribution(ax3, core_score, core_thr, bimodal_info)

    plt.suptitle(
        f'{protein_name} — Full Sequence Aggregation Mutability Landscape',
        fontsize=14, fontweight='bold', y=1.01
    )

    if output_path:
        _save_multiformat(fig, output_path, label="aggregation_mutability_landscape")

        # Figura standalone della sola distribuzione (per paper / supplementary)
        diag_path = str(Path(output_path).parent / 'hotspot_index_distribution.png')
        plot_sensitivity_distribution_standalone(
            core_score, core_thr, bimodal_info,
            protein_name=protein_name, output_path=diag_path
        )
    plt.close(fig)


def save_scan_report(scan_results, protein_name, out_dir):
    """Salva scan CSV: una riga per posizione, con top-3 mut compatibili e breaker."""
    rows = []
    delta_logp = scan_results['delta_logp']   # (L, 20)

    for i, (pos, wt_aa) in enumerate(zip(scan_results['positions'],
                                          scan_results['wt_aas'])):
        dlp    = delta_logp[i]
        wt_idx = AA_TO_IDX.get(wt_aa, -1)

        # Top-3 compatibili (esclude WT)
        sorted_desc = np.argsort(dlp)[::-1]
        compat = [(AA_ORDER[j], round(float(dlp[j]), 3))
                  for j in sorted_desc if j != wt_idx][:3]

        # Top-3 breaker
        sorted_asc = np.argsort(dlp)
        breakers   = [(AA_ORDER[j], round(float(dlp[j]), 3))
                      for j in sorted_asc if j != wt_idx][:3]

        mean_signed_i = float(scan_results['mean_signed'][i])
        is_core_i     = scan_results['core_score'][i] > np.percentile(scan_results['core_score'], 75)

        # Categoria esplicita: distingue hotspot "veri" (breaker, WT ottimale)
        # da posizioni dove il WT stesso appare sub-ottimale secondo il modello
        # (core_score alto solo perché quasi tutte le mutazioni sono FAVOREVOLI).
        if not is_core_i:
            category_i = 'neutral'
        elif mean_signed_i < 0:
            category_i = 'breaker_hotspot'      # vero hotspot: WT ottimale, mutazioni penalizzate
        else:
            category_i = 'tolerant_suboptimal_wt'  # WT stesso sub-ottimale, mutazioni favorite

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
    print(f"✓ Sequence scan CSV salvato: {csv_path}")
    return csv_path

def plot_delta_logp(delta_logp, column_labels, window_seq, wt_idx,
                   output_path=None, figsize=(18, 10), protein_name='Protein'):
    """
    Heatmap ΔlogP (20 AA × N finestre).
    Colormap divergente: blu = compatibile, rosso = breaker.
    """
    row_labels = [AA_NAMES[i] for i in range(20)]

    vmax = max(2.0, float(np.percentile(np.abs(delta_logp), 95)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    im = ax.imshow(delta_logp, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    # Griglia
    ax.set_xticks(np.arange(delta_logp.shape[1]) + 0.5, minor=True)
    ax.set_yticks(np.arange(20) + 0.5, minor=True)
    ax.grid(which='minor', color='#cccccc', linestyle='-', linewidth=0.4)
    ax.tick_params(which='minor', size=0)
    ax.set_xlim(-0.5, delta_logp.shape[1] - 0.5)
    ax.set_ylim(20 - 0.5, -0.5)

    # Valori nelle celle
    for c in range(delta_logp.shape[1]):
        for r in range(20):
            val = delta_logp[r, c]
            if abs(val) > 0.05:
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                        color=color, fontsize=10, fontweight='bold')
            else:
                # Valore neutro (include riga WT = 0.0 esatto)
                ax.text(c, r, '0.0', ha='center', va='center',
                        color='#999999', fontsize=9.5, fontstyle='italic')

    # Evidenzia riga WT
    ax.add_patch(plt.Rectangle(
        (-0.5, wt_idx - 0.5), delta_logp.shape[1], 1,
        fill=False, edgecolor='#FFD700', linewidth=2.5, zorder=3
    ))

    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha='right',
                   rotation_mode='anchor', fontsize=11)
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(
        [f'{"→ " if i == wt_idx else "   "}{AA_NAMES[i]}' for i in range(20)],
        fontsize=14
    )

    ax.set_xlabel(f'{protein_name} windows', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Mutation', fontsize=14, fontweight='bold')
    ax.set_title(
        f'ΔlogP Mutational Landscape — {protein_name}\n'
        f'Window: {window_seq}  |  WT center: {AA_ORDER[wt_idx]} (highlighted row)',
        fontsize=14, fontweight='bold', pad=15
    )


    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('ΔlogP  (mut − WT)', fontsize=14, fontweight='bold')
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
    Figura multi-pannello (4 righe):
      1. Entropy profile
      2. Core score (fibril conservation)
      3. Mutational tolerance
      4. WT probability vs finestra (controllo)
    """
    N    = len(entropy)
    x    = np.arange(N)
    xlbls = [f"W{i+1}" for i in range(N)]

    # Soglie per hotspot / core
    ent_thr   = np.percentile(entropy, 75)
    core_thr  = np.percentile(core_score, 75)

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs  = gridspec.GridSpec(4, 1, hspace=0.55, figure=fig)

    # ── Pannello 1: Entropy ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(x, entropy, color='#4A90D9', edgecolor='none', alpha=0.85)
    for i, (v, b) in enumerate(zip(entropy, bars)):
        if v > ent_thr:
            b.set_color('#4A90D9')
            b.set_alpha(0.95)
    ax1.axhline(ent_thr, color='#4A90D9', linestyle='--', lw=1.5,
                label=f'75° percentile ({ent_thr:.2f})')
    ax1.set_ylabel('Shannon Entropy', fontsize=10, fontweight='bold')
    ax1.set_title(f'Mutational entropy per window — {window_seq}',
                  fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlbls, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_xlim(-0.5, N - 0.5)

    # Annotazioni hotspot
    for i, v in enumerate(entropy):
        if v > ent_thr:
            ax1.text(i, v + 0.01, 'H', ha='center', va='bottom',
                     fontsize=7, color='#E8512B', fontweight='bold')

    # ── Pannello 2: Core score ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(x, core_score, color='#2E7D32', edgecolor='none', alpha=0.85)
    for i, (v, b) in enumerate(zip(core_score, bars2)):
        if v > core_thr:
            b.set_color('#2E7D32')
    ax2.axhline(core_thr, color='#2E7D32', linestyle='--', lw=1.5,
                label=f'75° percentile ({core_thr:.2f})')
    ax2.set_ylabel('Fibril Core Score\n(mean |ΔlogP|)', fontsize=10, fontweight='bold')
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

    # ── Pannello 3: Mutational tolerance ────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    bars3 = ax3.bar(x, tolerance, color='#7B1FA2', edgecolor='none', alpha=0.85)
    for i, (v, b) in enumerate(zip(tolerance, bars3)):
        if v >= 5:
            b.set_color('#FF6F00')
    ax3.axhline(5, color='#FF6F00', linestyle='--', lw=1.5,
                label='Min. 5 mutazioni tollerate (designable)')
    ax3.set_ylabel('N tolerated mutations\n(ΔlogP > -2)', fontsize=10, fontweight='bold')
    ax3.set_title('Mutational tolerance — designable positions',
                  fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlbls, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.set_xlim(-0.5, N - 0.5)

    for i, v in enumerate(tolerance):
        ax3.text(i, v + 0.1, str(int(v)), ha='center', va='bottom',
                 fontsize=7, color='black')

    # ── Pannello 4: WT probability ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    # ΔlogP del WT rispetto a se stesso = 0, ma usiamo la prob assoluta
    wt_delta = delta_logp[wt_idx, :]   # (N,) — 0 per definizione quando mascherato
    # Mostra invece il valore assoluto medio dei breaker per finestra
    n_breakers_per_win = (delta_logp < breaker_threshold).sum(axis=0)
    ax4.bar(x, n_breakers_per_win, color='#C62828', edgecolor='none', alpha=0.85)
    ax4.set_ylabel(f'N breaker mutations\n(ΔlogP < {breaker_threshold})',
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
    Scatter 2D degli embedding strutturali, colorato per entropy (sx)
    e core score (dx).
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
        f'Structural embedding space — {window_seq}  [{method_label}]',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    if output_path:
        _save_multiformat(fig, output_path, label="embedding_scatter")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCCO 5 — Report JSON / CSV
# ═══════════════════════════════════════════════════════════════════════════════

def build_report(delta_logp, entropy, core_score, tolerance,
                 column_labels, window_seq, wt_idx,
                 breaker_threshold=-5.0, design_threshold=-1.0):
    """
    Genera dict con riepilogo numerico completo.
    """
    center = len(window_seq) // 2
    wt_aa  = window_seq[center]

    breakers, _  = find_breakers(delta_logp, threshold=breaker_threshold, percentile=None)
    designable = find_designable_windows(delta_logp, design_threshold)
    hotspots   = np.where(entropy > np.percentile(entropy, 75))[0].tolist()
    core_wins  = np.where(core_score > np.percentile(core_score, 75))[0].tolist()

    # Per ogni finestra, top-3 mutazioni più compatibili e top-3 breaker
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
        'protein':   'α-synuclein',
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
    """Salva JSON e CSV del report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / 'analysis_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report JSON salvato: {json_path}")

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
    print(f"✓ Report CSV salvato: {csv_path}")

    # Stampa breaker peggiori
    print("\n── Top-10 breaker mutations ──")
    print(f"{'Win':>5} {'AA':>4} {'ΔlogP':>10}")
    print("-" * 25)
    for win_idx, aa, score in report['worst_breakers']:
        print(f"  {win_idx:>3}   {aa:>2}   {score:>9.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Heatmap probabilità + analisi mutazionale completa α-synuclein"
    )
    # Input
    ap.add_argument('--model',            required=True)
    ap.add_argument('--data',             required=True)
    ap.add_argument('--protein_metadata', required=True)
    ap.add_argument('--rotamer_csv',      default='rotamer_localmax_maxdens.csv')
    ap.add_argument('--fasta',            default=None)
    ap.add_argument('--window_seq',       default='NVGGA')
    ap.add_argument('--protein_name',     default='α-synuclein')

    # Tokens
    ap.add_argument('--mask_token_aa', type=int, default=20)
    ap.add_argument('--mask_token_cl', type=int, default=0)

    # Output
    ap.add_argument('--output',   default='results/asynuclein_analysis',
                    help='Cartella di output (verrà creata se non esiste)')

    # Analisi
    ap.add_argument('--breaker_threshold', type=float, default=None,
                    help='Soglia FISSA ΔlogP per definire breaker. '
                         'Se omessa, usa --breaker_percentile (raccomandato).')
    ap.add_argument('--breaker_percentile', type=float, default=5.0,
                    help='Percentile per soglia breaker adattiva (default: 5 = bottom 5%%). '
                         'Ignorato se --breaker_threshold è fornita.')
    ap.add_argument('--design_threshold',  type=float, default=-1.0,
                    help='Soglia ΔlogP per definire mutazioni tollerate (default: -1)')
    ap.add_argument('--min_tolerated',     type=int,   default=5,
                    help='Min mutazioni tollerate per finestra designable (default: 5)')
    ap.add_argument('--embedding_method',  default='pca',
                    choices=['pca', 'umap'],
                    help='Metodo riduzione embedding (default: pca)')
    ap.add_argument('--skip_embedding',    action='store_true',
                    help='Salta estrazione embedding (più veloce)')
    ap.add_argument('--sequence_scan',     action='store_true',
                    help='Esegui scan su TUTTA la sequenza (richiede --fasta)')
    ap.add_argument('--start_residue',    type=int, default=1,
                    help='Residuo di partenza in numerazione CANONICA (default: 1 = nessun filtro). '
                         'Interpretato secondo --residue_offset. '
                         'Finestre e posizioni prima di questo residuo vengono escluse.')
    ap.add_argument('--end_residue',      type=int, default=None,
                    help='Residuo finale in numerazione CANONICA (default: None = nessun filtro). '
                         'Interpretato secondo --residue_offset. '
                         'Finestre e posizioni dopo questo residuo vengono escluse.')
    ap.add_argument('--residue_offset',   type=int, default=0,
                    help='Offset per convertire la posizione 1-based del FASTA nella numerazione '
                         'canonica del residuo: residuo_canonico = posizione_FASTA + offset. '
                         'Serve quando il FASTA del costrutto contiene Met/His-tag/linker aggiunti. '
                         'Es.: HET-s 2RNM -> --residue_offset 216 (FASTA pos 2 = K218); '
                         'Mrip3/RIPK3 6JPD -> --residue_offset 402 (FASTA pos 7 = res 409). '
                         'Vale sia per le etichette mostrate sia per --start_residue/--end_residue.')
    ap.add_argument('--auto_offset',      action='store_true',
                    help='Determina --residue_offset in automatico dal PDB id (header del FASTA) '
                         'via mappatura SIFTS di PDBe, usando la numerazione depositata (autore). '
                         'Richiede rete (ebi.ac.uk). Un --residue_offset esplicito (!=0) ha priorità. '
                         'Se il PDB non è mappabile, si usa offset 0 con un warning.')

    # Layout
    ap.add_argument('--colormap',  default='RdPu')
    ap.add_argument('--figsize',   nargs=2, type=float, default=[18, 10], metavar=('W', 'H'))
    ap.add_argument('--cpu',       action='store_true')

    args   = ap.parse_args()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Carica modello ──
    model = load_model(args.model, device)

    # ── 2. Carica dataset ──
    data, protein_indices = load_dataset(
        args.data, args.protein_metadata, args.protein_name
    )
    if len(protein_indices) == 0:
        print(f"⛔  Nessuna finestra trovata per '{args.protein_name}'.")
        sys.exit(1)

    # ── 3. Trova finestre matching ──
    matched_indices, _ = find_matching_windows(
        data, protein_indices, args.window_seq
    )
    if len(matched_indices) == 0:
        print(f"⛔  Nessuna finestra matcha '{args.window_seq}'.")
        sys.exit(1)

    # ── 4. FASTA ──
    fasta_seq = None
    if args.fasta and Path(args.fasta).exists():
        fasta_seq = read_fasta(args.fasta)
        print(f"\nFASTA: {len(fasta_seq)} residui")
    elif args.sequence_scan:
        print("⛔  --sequence_scan richiede --fasta. Sequence scan disabilitato.")
        args.sequence_scan = False

    # ── 4a. Offset di numerazione automatico (opzionale) ──
    #   residuo_canonico = posizione_FASTA_1based + residue_offset
    #   --residue_offset esplicito (!=0) vince; altrimenti prova SIFTS se --auto_offset.
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
                  f"(residuo_canonico = pos_FASTA + {off})")
        else:
            print(f"  [auto_offset] non determinato per {pdb_for_offset}; "
                  f"uso residue_offset = {args.residue_offset}.")
    elif args.auto_offset and args.residue_offset != 0:
        print(f"  [auto_offset] ignorato: --residue_offset={args.residue_offset} impostato manualmente.")

    # ── 4b. Filtra finestre per start_residue ──
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
            print(f"\n  [start_residue={args.start_residue}] Escluse {n_removed} finestre "
                  f"con residuo centrale prima del residuo {args.start_residue}.")
        matched_indices = filtered_matched
        print(f"  Finestre mantenute (≥ residuo {args.start_residue}): {len(matched_indices)}")

        if len(matched_indices) == 0:
            print(f"⛔  Nessuna finestra rimasta dopo filtro start_residue={args.start_residue}.")
            sys.exit(1)

    # ── 4c. Filtra finestre per end_residue ──
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
            print(f"\n  [end_residue={args.end_residue}] Escluse {n_removed} finestre "
                  f"con residuo centrale dopo il residuo {args.end_residue}.")
        matched_indices = filtered_matched
        print(f"  Finestre mantenute (≤ residuo {args.end_residue}): {len(matched_indices)}")

        if len(matched_indices) == 0:
            print(f"⛔  Nessuna finestra rimasta dopo filtro end_residue={args.end_residue}.")
            sys.exit(1)

    # ── 5. Matrice probabilità ──
    prob_matrix = compute_probability_matrix(
        model, data, matched_indices, device,
        mask_token_aa=args.mask_token_aa,
        mask_token_cl=args.mask_token_cl
    )

    # ── 6. Etichette ──
    with open(args.protein_metadata) as f:
        prot_meta_full = json.load(f)

    column_labels = build_column_labels(
        matched_indices, data, fasta_seq, args.window_seq,
        protein_metadata=prot_meta_full, residue_offset=args.residue_offset
    )

    # ── 7. Heatmap probabilità (originale) ──
    print_summary(prob_matrix, column_labels)
    plot_heatmap(
        prob_matrix, column_labels, args.window_seq,
        output_path=str(out_dir / 'heatmap_prob.png'),
        figsize=tuple(args.figsize), colormap=args.colormap,
        protein_name=args.protein_name
    )

    # ── 8. ΔlogP landscape ──
    delta_logp, wt_idx = compute_delta_logp(prob_matrix, args.window_seq)
    plot_delta_logp(
        delta_logp, column_labels, args.window_seq, wt_idx,
        output_path=str(out_dir / 'heatmap_delta_logp.png'),
        figsize=tuple(args.figsize),
        protein_name=args.protein_name
    )

    # ── 9. Metriche per finestra ──
    entropy    = compute_entropy_profile(prob_matrix)
    core_score = compute_core_score(delta_logp)
    tolerance  = compute_mutational_tolerance(delta_logp, threshold=-2.0)

    # Determina soglia breaker: percentile adattivo (raccomandato) o fissa
    if args.breaker_threshold is not None:
        effective_threshold = args.breaker_threshold
        breakers, _  = find_breakers(delta_logp, threshold=effective_threshold, percentile=None)
    else:
        breakers, effective_threshold = find_breakers(
            delta_logp, percentile=args.breaker_percentile
        )

    # ── analysis_summary, embedding e bimodal threshold disabilitati ──
    is_true_emb = False

    # ── 11. Report finestre ──
    report = build_report(
        delta_logp, entropy, core_score, tolerance,
        column_labels, args.window_seq, wt_idx,
        breaker_threshold=effective_threshold,
        design_threshold=args.design_threshold
    )
    save_report(report, out_dir)

    # ── 12. Sequence scan (opzionale, figura da paper) ──
    if args.sequence_scan and fasta_seq:
        print(f"\n{'='*60}")
        print("SEQUENCE SCAN — Aggregation Mutability Landscape")
        print(f"  Scansione di {len(fasta_seq)} residui di {args.protein_name}...")
        print(f"  Finestre nel dataset: {len(protein_indices)}")
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

    # ── Summary finale ──
    print(f"\n{'='*60}")
    print(f"✓ Analisi completa salvata in: {out_dir}")
    print(f"  heatmap_prob.png          ← probabilità (originale)")
    print(f"  heatmap_delta_logp.png    ← landscape ΔlogP [fig. principale]")
    print(f"  analysis_report.json      ← riepilogo numerico")
    print(f"  analysis_report.csv       ← tabella per finestra")
    if args.sequence_scan:
        print(f"  aggregation_mutability_landscape.png  ← FIGURA DA PAPER (3 pannelli)")
        print(f"  hotspot_index_distribution.png ← distribuzione bimodale standalone")
        print(f"  sequence_scan_report.csv              ← landscape tabulare")
    print(f"\n  Soglia breaker usata: {effective_threshold:.3f} "
          f"({'p' + str(args.breaker_percentile) + ' adattivo' if args.breaker_threshold is None else 'fissa'})")
    if args.start_residue > 1:
        print(f"  Filtro start_residue: ≥ {args.start_residue}")
    if args.end_residue is not None:
        print(f"  Filtro end_residue:   ≤ {args.end_residue}")
    print(f"{'='*60}")
    print("✓ Done!")


if __name__ == '__main__':
    main()