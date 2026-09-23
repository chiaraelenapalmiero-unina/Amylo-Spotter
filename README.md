# Amylo-spotter

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

<img align="left" hspace="15" width="297" alt="amylospotter" src="https://github.com/user-attachments/assets/02db36b0-b693-4029-917d-ce5665b11b20" />

Amylo-spotter is a BiLSTM-based tool for identifying structural hotspots in amyloid fibril proteins. Given a 5-residue structural window (backbone position, rotamer, secondary structure, solvent exposure), the model learns to predict the central residue's identity and side-chain conformation, and the resulting per-position confidence is used to flag mutation-sensitive "hotspots" along a fibril structure.

Manuscript in progress: *Palmiero et al.*

<br clear="left"/>

### **Website version available at this link: ...**

## Getting started

Clone the repository:

```bash
git clone https://github.com/chiaraelenapalmiero-unina/Amylo-Spotter.git
cd Amylo-Spotter
```

If you don't use git, you can also download the repository as a
[ZIP archive](https://github.com/chiaraelenapalmiero-unina/Amylo-Spotter/archive/refs/heads/main.zip).

## Environments

The project uses two separate **conda** environments:

| Environment         | Python | Used for |
|---------------------|--------|----------|
| `dssp_env`          | 3.8    | DSSP only (`data_pipeline/AUTOMATED_DSSP.py`, `data_pipeline/dssp.py`) |
| `amylo_spotter_env` | 3.13   | everything else: window construction, training, post-processing (CUDA 12.4, PyTorch 2.5.1) |

To create them:

```bash
conda env create -f envs/dssp_env.yml
conda env create -f envs/amylo_spotter_env.yml
```

and activate the one you need with:

```bash
conda activate amylo_spotter_env
```

## Data

The tokenized training data are included in the repository as `data/windows_all_local.npz`.

The file contains the 5-residue structural windows extracted from **all fibril structures
in the dataset. Each window can be traced back to its source structure, protein and class
through the files in `metadata/`:

| File | Content |
|------|---------|
| `proteins_metadata_with_class.json` | metadata for every structure in the dataset, including its protein class |
| `pdb_to_protein_class.csv` | mapping from PDB ID to protein and class |

The post-processing scripts use these files to select the windows of a given protein
(`--protein_name` together with `--protein_metadata`).

## Trained models

Trained models are too large for GitHub and are available on Zenodo:
[10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

| File | Content |
|------|---------|
| `trained_models_generic.zip` | model trained across all protein classes |
| `trained_models_per_protein.zip` | one model per individual protein |

Download the archives and extract them into the repository root:

```bash
unzip trained_models_generic.zip -d trained_models/
unzip trained_models_per_protein.zip -d trained_models/
```

## Quick start

Compute the aggregation mutability landscape of α-synuclein with its protein-specific
model (after downloading the trained models from Zenodo, see above). Run from the
repository root:

```bash
conda activate amylo_spotter_env
python postprocessing/mutation_heatmap_extended_correction.py \
    --model trained_models/teacher_α-synuclein"/best_teacher.pt \
    --data data/windows_all_local.npz \
    --protein_metadata metadata/proteins_metadata_with_class.json \
    --rotamer_csv common/rotamer_localmax_maxdens.csv \
    --fasta examples/aS_7v49.fasta \
    --protein_name "α-synuclein" \
    --sequence_scan --start_residue 38 --end_residue 95 \
    --output results/asynuclein
```

The main output is `results/asynuclein/aggregation_mutability_landscape.png`:

<p align="center">
  <img width="100%" alt="aggregation_mutability_landscape" src="https://github.com/user-attachments/assets/dc08f8d0-bcf7-4efd-88b8-b1427e7533a0" />
</p>

The figure has three panels:

- **Top — mutability landscape:** ΔlogP = log p(mut) − log p(WT) for every possible
  substitution (rows) at each sequence position (columns); black dots mark the wild-type
  residue. Red cells are substitutions the model disfavors relative to the wild type,
  blue cells are favored ones.
- **Middle — hotspot index:** mean |ΔlogP| per position. Positions above the threshold
  (dashed line) are flagged as mutation-sensitive hotspots.
- **Bottom — hotspot index distribution:** the threshold is set automatically from this
  distribution. The histogram of hotspot index values is smoothed with a Gaussian filter,
  and the threshold is the first point after the dominant peak where the density drops
  below 5% of the peak height. Positions to its right form the tail of rigid,
  mutation-sensitive residues. If the density never drops that low, the 75th percentile
  is used instead.

The analysis only covers proteins whose structural windows are included in the dataset.

## Pipeline overview

The project is organized around three stages:

1. **Data pipeline** — start from PDB structures, run DSSP to get secondary structure and
   solvent accessibility, classify residues as buried/exposed, assign each residue to a
   rotamer cluster, and slide a 5-residue window along each β-strand to build the tokenized
   training arrays.
2. **Model** — a BiLSTM trained as a masked-residue predictor (amino acid identity + rotamer
   cluster), with k-fold cross-validation and a leave-one-protein-class-out scheme to avoid
   data leakage across structures of the same protein.
3. **Post-processing** — use a trained model to score point mutations across a
   window/sequence and produce mutational landscapes, entropy profiles, and hotspot heatmaps.

## How to run

> **Note:** all scripts should be run from the repository root, since some of them add
> `common/` and `model/` to `sys.path` at import time.

### 1. Data pipeline

Within `dssp_env`:

```bash
python data_pipeline/AUTOMATED_DSSP.py ...
```

Within `amylo_spotter_env`:

```bash
python data_pipeline/classification_exposition.py ...
python data_pipeline/window_array_from_json_to_csv.py ...
python data_pipeline/build_windows5_tokens.py ...
python data_pipeline/convert_global_to_local_clusters.py ...
```

### 2. Training

```bash
python model/training/teacher_kfold_train_complete.py ...
```

### 3. Post-processing

```bash
python postprocessing/mutation_heatmap_extended_correction.py ...
```

## Repository structure

```
Amylo-Spotter/
├── envs/                          # conda environments
│   ├── dssp_env.yml               # Python 3.8 — DSSP only
│   └── amylo_spotter_env.yml      # Python 3.13, PyTorch/CUDA — everything else
│
├── common/                        # shared code, used by both data_pipeline and model
│   ├── rotamer_library.py         # AA <-> rotamer cluster mappings
│   └── rotamer_localmax_maxdens.csv
│
├── data/
│   └── windows_all_local.npz      # tokenized windows, pathological + functional fibrils
│
├── data_pipeline/                 # PDB structures -> tokenized training windows
│   ├── dssp.py                    # minimal DSSP usage example
│   ├── AUTOMATED_DSSP.py          # batch DSSP + ACC normalization over a folder of PDBs
│   ├── classification_exposition.py        # buried/exposed classification from ACC_norm
│   ├── window_array_from_json_to_csv.py    # builds 5-residue windows (χ-angles, SS, SASA)
│   ├── build_windows5_tokens.py            # windows JSON -> tokenized .npz arrays
│   ├── convert_global_to_local_clusters.py # global -> per-residue rotamer cluster indices
│   └── residues_distribution.py            # KDE of solvent-accessibility distribution
│
├── model/                         # architecture, dataset, and training
│   ├── dataset.py                 # Windows3Dataset (masked-residue dataset)
│   ├── topk_model2_FULL.py        # ProteinMLM_v2 — full training
│   ├── topk_model2_LOO.py         # ProteinMLM_v2 — leave-one-out training on individual
│   │                              # pathological fibrils
│   ├── create_balanced_testset.py # builds an exposure/AA-balanced held-out test set
│   ├── training/                  # k-fold training (leave-one-protein-class-out)
│   └── Training_full/             # full training runs
│
├── metadata/                      # maps each structure to its protein and class
│   ├── proteins_metadata_with_class.json
│   ├── proteins_metadata_pathological.json
│   └── pdb_to_protein_class.csv
│
├── postprocessing/
│   └── mutation_heatmap_extended_correction.py  # mutation scoring, hotspot analysis
│
├── trained_models/                # (not tracked) download from Zenodo, see above
│
├── .gitignore
└── README.md
```

## How to cite

If you use Amylo-spotter, please cite:

> Palmiero et al. *Manuscript in preparation.*

Trained models: Zenodo, DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

## Contact

For any questions/comments (**_regarding this code_**) please contact:
chiaraelena.palmiero@unina.it

## TODO

- [ ] Upload trained models to Zenodo and update the DOI
- [ ] Add a `LICENSE`
