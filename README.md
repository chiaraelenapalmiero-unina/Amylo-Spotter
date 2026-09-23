
# Amylo-spotter
<img align="left"  hspace="15"  width="297" height="165" alt="amylospotter" src="https://github.com/user-attachments/assets/02db36b0-b693-4029-917d-ce5665b11b20" />





Amylo-spotter is a BiLSTM-based tool for identifying structural hotspots in amyloid fibril proteins. Given a 5-residue structural window (backbone position, rotamer, secondary structure, solvent exposure), the model learns to predict the central residue's identity and side-chain conformation, and the resulting per-position confidence is used to flag mutation-sensitive "hotspots" along a fibril structure.

Manuscript in progress: *Palmiero et al.*

<br clear="left"/>

### **Website version available at this link: ...**


## Pipeline overview

The project is organized around three stages:

1. **Data pipeline** — start from PDB structures, run DSSP to get secondary
   structure and solvent accessibility, classify residues as buried/exposed,
   assign each residue to a rotamer cluster, and slide a 5-residue window
   along each β-strand to build the tokenized training arrays.
2. **Model** — a BiLSTM trained as a masked-residue predictor
   (amino acid identity + rotamer cluster), with k-fold
   cross-validation and a leave-one-protein-class-out scheme to avoid
   data leakage across structures of the same protein.
4. **Post-processing** — use a trained model to score point mutations
   across a window/sequence and produce mutational landscapes, entropy
   profiles, and hotspot heatmaps.

## Repository structure

```
amylo-spotter/
├── envs/                        # conda environments
│   ├── dssp_env.yml             # Python 3.8 — DSSP only
│   └── amylo_spotter_env.yml    # Python 3.13, PyTorch/CUDA — everything else
│
├── common/                      # shared code, used by both data_pipeline and model
│   ├── rotamer_library.py       # AA <-> rotamer cluster mappings
│   └── rotamer_localmax_maxdens.csv
│
├── data_pipeline/                # PDB structures -> tokenized training windows
│   ├── dssp.py                   # minimal DSSP usage example
│   ├── AUTOMATED_DSSP.py         # batch DSSP + ACC normalization over a folder of PDBs
│   ├── classification_exposition.py   # buried/exposed classification from ACC_norm
│   ├── window_array_from_json_to_csv.py  # builds 5-residue windows (χ-angles, SS, SASA)
│   ├── build_windows5_tokens.py  # windows JSON -> tokenized .npz arrays
│   ├── convert_global_to_local_clusters.py  # global -> per-residue rotamer cluster indices
│   └── residues_distribution.py  # KDE of solvent-accessibility distribution
│
├── model/                        # architecture, dataset, and training
│   ├── dataset.py                 # Windows3Dataset (masked-residue dataset)
│   ├── topk_model2.py             # ProteinMLM_v2 — optimized (vectorized/einsum) architecture
│   ├── topk_model2_fib.py         # same architecture, standalone script used for
│   │                                leave-one-out training on individual pathological fibrils
│   ├── create_balanced_testset.py # builds an exposure/AA-balanced held-out test set
│   ├── plot_training_curves.py    # training/validation diagnostics across folds
│   └── training/
│       ├── teacher_kfold_train_complete.py  # main training entry point
│       │                                      (k-fold, leave-one-protein-class-out)
│       └── train_globular_vs_fibrils.py     # full training run, globular + fibril proteins
│
├── trained_models/
│   ├── generic/                  # single model trained across all protein classes
│   └── per_protein/              # one model per individual protein
│
├── metadata/
│   ├── proteins_metadata_with_class.json
│   ├── proteins_metadata_pathological.json
│   └── pdb_to_protein_class.csv
│
├── postprocessing/
    └── mutation_heatmap_extended_correction.py   # mutation scoring, entropy/hotspot analysis
```

## Environments

```bash
conda env create -f envs/dssp_env.yml
conda env create -f envs/amylo_spotter_env.yml
```

`dssp_env` (Python 3.8) is only needed to run DSSP
(`data_pipeline/AUTOMATED_DSSP.py`, `data_pipeline/dssp.py`).
Everything else — window construction, training, post-processing — runs
under `amylo_spotter_env` (Python 3.13, CUDA 12.4, PyTorch 2.5.1).


## TODO

- [ ] Populate `trained_models/` with checkpoints (generic + per-protein)
- [ ] Add a `LICENSE`
