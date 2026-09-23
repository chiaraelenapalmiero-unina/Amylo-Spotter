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

## Data and trained models

Trained models and preprocessed training data are too large for GitHub and are
available on Zenodo: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

The record contains:

| File | Content |
|------|---------|
| `trained_models_generic.zip` | model trained across all protein classes |
| `trained_models_per_protein.zip` | one model per individual protein |
| `training_data_npz.zip` | tokenized 5-residue windows (`.npz`) |

Download the archives and extract them into the repository root:

```bash
unzip trained_models_generic.zip -d trained_models/
unzip trained_models_per_protein.zip -d trained_models/
unzip training_data_npz.zip -d data/
```

## Quick start

Predict hotspots on a single fibril structure using the generic model
(after downloading it from Zenodo, see above):

```bash
conda activate amylo_spotter_env
python postprocessing/mutation_heatmap_extended_correction.py \
    --pdb examples/example_fibril.pdb \
    --model trained_models/generic/<model_file>.pt \
    --out results/
```

This produces a per-residue mutational landscape and a hotspot heatmap
in `results/`:

<p align="center">
  <img src="docs/images/example_heatmap.png" alt="Example hotspot heatmap" width="700">
</p>

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
├── metadata/
│   ├── proteins_metadata_with_class.json
│   ├── proteins_metadata_pathological.json
│   └── pdb_to_protein_class.csv
│
├── postprocessing/
│   └── mutation_heatmap_extended_correction.py  # mutation scoring, entropy/hotspot analysis
│
├── trained_models/                # (not tracked) download from Zenodo, see above
├── data/                          # (not tracked) download from Zenodo, see above
│
├── .gitignore
└── README.md
```

## How to cite

If you use Amylo-spotter, please cite:

> Palmiero et al. *Manuscript in preparation.*

Trained models and data: Zenodo, DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

## Contact

For any questions/comments (**_regarding this code_**) please contact:
chiaraelena.palmiero@unina.it

## TODO

- [ ] Upload trained models and `.npz` data to Zenodo and update the DOI
- [ ] Add a quick-start example (single PDB -> hotspot heatmap)
- [ ] Add an example output figure
- [ ] Add a `LICENSE`
