"""
TEACHER KFOLD TRAIN - VERSION WITH PROTEIN_CLASS
=================================================
Complete script for training the teacher model, with:

  1. TeacherModel (ProteinMLM_v2 + aggregation_head)
  2. Training loss: loss_aa + lambda_cl * loss_cl + lambda_agg * MSE(agg_score, 1-sasa_center)
     (theoretically should be like this: loss_aa + lambda_cl * loss_cl)
  3. Split with leave-one-PROTEIN-CLASS-out + exposed/buried stratification
  4. Checkpoint saving with metadata for generate_teacher_predictions.py

NEW: Uses protein_class instead of PDB ID for the leave-one-out split!
    - Excludes ALL structures of the same protein (e.g. every Tau PDB,
      every alpha-synuclein PDB, etc.)
    - Avoids data leakage when the same protein has multiple PDB entries
    - Uses the real protein names (Tau, Amyloid-beta, alpha-synuclein, etc.)

Usage:
    # First build the metadata file with protein_class
    python create_metadata_from_npz_csv.py \
        --npz windows_all_local.npz \
        --csv windows_all.csv \
        --output proteins_metadata_with_class.json
    
  python teacher_kfold_train_full.py \
    --data windows_all_local.npz \
    --test_data test.npz \ (to be changed according to your own testset)
    --run_name full_model \
    --rotamer_csv rotamer_localmax_maxdens.csv \
    --outdir teachers_full/ \
    --validation_mode all \
    --k_folds 5 \
    --epochs 200 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-3 \
    --grad_clip 1.0 \
    --d_embed 256 \
    --d_hidden 512 \
    --lstm_layers 3 \
    --dropout 0.25 \
    --lambda_cl 1.0 \
    --patience 35 \
    --embed_dropout 0.3 \
    --label_smoothing 0.08 \
    --use_mixup \
    --mixup_alpha 0.2 \
    --sasa_threshold 0.123

Examples of valid protein_class values:
    - Tau (2035 windows)
    - alpha-synuclein (1234 windows)
    - Prion Protein (837 windows)
    - Amyloid-beta(1-42) (501 windows)
    - SOD1 (411 windows)
    - etc.
    
   
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import shutil
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import time

# Import from topk_model2 (shared functions)
import sys
# topk_model2.py / dataset.py live in ../ (model/), rotamer_library.py in ../../common
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "common"))

from topk_model2 import (
    ProteinMLM_v2,
    WarmupCosineScheduler,
    train_epoch_v2,
    eval_epoch_v2
)
from dataset import Windows3Dataset, collate_fn
from rotamer_library import load_rotamer_library

# Import of TeacherModel and the aggregation target
# ProteinMLM_v2 is already imported from topk_model2 above


# ============================================================
# EVAL EPOCH (no aggregation head - identical to kfold_train.py)
# ============================================================
@torch.no_grad()
def eval_epoch_teacher(model, loader, aa_to_n_clusters, device='cuda', ss_names=None,
                       lambda_cl=1.0, label_smoothing=0.08):
    """
    Runs one evaluation epoch for the teacher model (no aggregation
    head involved) by delegating directly to `eval_epoch_v2`.

    Args:
        model: The model to evaluate.
        loader: DataLoader providing the evaluation batches.
        aa_to_n_clusters: Dict mapping amino acid index to its number of
            rotamer clusters (used to size/mask the cluster logits).
        device: Torch device to run evaluation on (default: 'cuda').
        ss_names: Optional dict mapping SS class index to a display name,
            used for the per-secondary-structure breakdown.
        lambda_cl: Weight applied to the cluster loss term.
        label_smoothing: Label smoothing factor used in the loss.

    Returns:
        Whatever `eval_epoch_v2` returns: a dict of evaluation metrics
        (losses, top-k accuracies, optionally per-SS statistics).
    """
    return eval_epoch_v2(
        model, loader, aa_to_n_clusters,
        lambda_cl=lambda_cl, device=device,
        ss_names=ss_names, verbose_ss=False
    )


# TRAIN EPOCH (no aggregation head - identical to kfold_train.py)
# ============================================================
def train_epoch_teacher(model, loader, opt, aa_to_n_clusters,
                        lambda_cl=1.0, grad_clip=1.0,
                        device='cuda', label_smoothing=0.08,
                        use_mixup=False, mixup_alpha=0.2, ss_names=None):
    """
    Runs one training epoch for the teacher model (no aggregation head
    involved) by delegating directly to `train_epoch_v2`.

    Args:
        model: The model being trained.
        loader: DataLoader providing the training batches.
        opt: Optimizer used to update the model's parameters.
        aa_to_n_clusters: Dict mapping amino acid index to its number of
            rotamer clusters.
        lambda_cl: Weight applied to the cluster loss term.
        grad_clip: Max gradient norm for gradient clipping.
        device: Torch device to train on (default: 'cuda').
        label_smoothing: Label smoothing factor used in the loss.
        use_mixup: Whether to apply mixup data augmentation.
        mixup_alpha: Alpha parameter of the mixup Beta distribution.
        ss_names: Optional dict mapping SS class index to a display name.

    Returns:
        Whatever `train_epoch_v2` returns: a dict of training metrics for
        this epoch.
    """
    return train_epoch_v2(
        model, loader, opt, aa_to_n_clusters,
        lambda_cl=lambda_cl, grad_clip=grad_clip, device=device,
        label_smoothing=label_smoothing,
        use_mixup=use_mixup, mixup_alpha=mixup_alpha,
        ss_names=ss_names
    )


# ============================================================
# K-FOLD SPLIT with leave-one-protein-out
# ============================================================
def create_kfold_splits(dataset, k_folds=5, seed=42, validation_mode='all', 
                        sasa_threshold=0.1199, protein_metadata=None, 
                        leave_out_protein=None):
    """
    Creates K-fold splits stratified by exposed/buried.
    
    NEW: If protein_metadata and leave_out_protein are provided, that
    protein is first excluded from the pool (legacy leave-one-out for
    KD), and the exposed/buried stratification is then applied to the
    remaining folds.
    
    Args:
        dataset: Windows3Dataset or Subset (already filtered).
        k_folds: Number of folds.
        seed: Random seed.
        validation_mode: 'all', 'exposed', or 'buried'.
        sasa_threshold: SASA threshold used to define "exposed".
        protein_metadata: (OPTIONAL - legacy) dict {window_idx: protein_id}.
        leave_out_protein: (OPTIONAL - legacy) name of the protein to
            exclude.
    
    Returns:
        List[(train_indices, val_indices)] for each fold.
    """
    # NOTE: If protein_metadata and leave_out_protein are None, the
    # dataset is assumed to already be filtered (new behavior).
    
    if protein_metadata is not None and leave_out_protein is not None:
        # Legacy behavior: filter here
        valid_indices = set()
        for idx in range(len(dataset)):
            prot = protein_metadata.get(str(idx), "unknown")
            if prot != leave_out_protein:
                valid_indices.add(idx)
        
        print(f"\n[LEAVE-ONE-PROTEIN-OUT (legacy mode)]")
        print(f"  Excluded protein: {leave_out_protein}")
        print(f"  Original pool: {len(dataset)} windows")
        print(f"  Pool after exclusion: {len(valid_indices)} windows")
    else:
        # New behavior: use the whole dataset (already filtered)
        valid_indices = set(range(len(dataset)))

    # Exposed/buried stratification over the valid pool
    exposed_indices = []
    buried_indices = []
    
    for i in range(len(dataset)):
        if i not in valid_indices:  # skip excluded windows
            continue
        
        sample = dataset[i]
        center_idx = sample['aa'].shape[0] // 2
        center_sasa = sample['sasa'][center_idx].item()
        is_exposed = center_sasa >= sasa_threshold
        
        if is_exposed:
            exposed_indices.append(i)
        else:
            buried_indices.append(i)
    
    exposed_indices = np.array(exposed_indices, dtype=np.int64)
    buried_indices = np.array(buried_indices, dtype=np.int64)
    
    print(f"\n[K-FOLD SPLIT INFO - SASA threshold={sasa_threshold}]")
    print(f"  Exposed centers (SASA>={sasa_threshold}): {len(exposed_indices)}")
    print(f"  Buried centers (SASA<{sasa_threshold}): {len(buried_indices)}")
    print(f"  K folds: {k_folds}")
    print(f"  Validation mode: {validation_mode}")
    
    rng = np.random.RandomState(seed)
    folds = []
    
    if validation_mode == 'exposed':
        rng.shuffle(exposed_indices)
        fold_size = len(exposed_indices) // k_folds
        
        for fold_idx in range(k_folds):
            start = fold_idx * fold_size
            end = start + fold_size if fold_idx < k_folds - 1 else len(exposed_indices)
            
            val_indices = exposed_indices[start:end]
            train_exposed = np.concatenate([
                exposed_indices[:start],
                exposed_indices[end:]
            ])
            train_indices = np.concatenate([train_exposed, buried_indices])
            rng.shuffle(train_indices)
            
            folds.append((train_indices, val_indices))
            print(f"  Fold {fold_idx+1}: Train={len(train_indices)}, Val={len(val_indices)} (exposed only)")
    
    elif validation_mode == 'buried':
        rng.shuffle(buried_indices)
        fold_size = len(buried_indices) // k_folds
        
        for fold_idx in range(k_folds):
            start = fold_idx * fold_size
            end = start + fold_size if fold_idx < k_folds - 1 else len(buried_indices)
            
            val_indices = buried_indices[start:end]
            train_buried = np.concatenate([
                buried_indices[:start],
                buried_indices[end:]
            ])
            train_indices = np.concatenate([train_buried, exposed_indices])
            rng.shuffle(train_indices)
            
            folds.append((train_indices, val_indices))
            print(f"  Fold {fold_idx+1}: Train={len(train_indices)}, Val={len(val_indices)} (buried only)")
    
    else:  # validation_mode == 'all'
        rng.shuffle(exposed_indices)
        rng.shuffle(buried_indices)
        
        exposed_fold_size = len(exposed_indices) // k_folds
        buried_fold_size = len(buried_indices) // k_folds
        
        for fold_idx in range(k_folds):
            exp_start = fold_idx * exposed_fold_size
            exp_end = exp_start + exposed_fold_size if fold_idx < k_folds - 1 else len(exposed_indices)
            val_exposed = exposed_indices[exp_start:exp_end]
            
            bur_start = fold_idx * buried_fold_size
            bur_end = bur_start + buried_fold_size if fold_idx < k_folds - 1 else len(buried_indices)
            val_buried = buried_indices[bur_start:bur_end]
            
            val_indices = np.concatenate([val_exposed, val_buried])
            rng.shuffle(val_indices)
            
            train_exposed = np.concatenate([
                exposed_indices[:exp_start],
                exposed_indices[exp_end:]
            ])
            train_buried = np.concatenate([
                buried_indices[:bur_start],
                buried_indices[bur_end:]
            ])
            train_indices = np.concatenate([train_exposed, train_buried])
            rng.shuffle(train_indices)
            
            folds.append((train_indices, val_indices))
            
            exp_ratio = len(val_exposed) / len(val_indices) * 100 if len(val_indices) > 0 else 0
            print(f"  Fold {fold_idx+1}: Train={len(train_indices)}, "
                  f"Val={len(val_indices)} ({len(val_exposed)} exp, {len(val_buried)} bur, "
                  f"{exp_ratio:.1f}% exposed)")
    
    return folds


# ============================================================
# TRAIN SINGLE FOLD (with teacher model)
# ============================================================
def train_single_fold(fold_idx, train_indices, val_indices, 
                      dataset, test_loader, args, device, ss_names, run_name):
    """
    Trains a single fold of the teacher model.
    
    DIFFERENCES from kfold_train.py:
      1. Uses TeacherModel instead of plain ProteinMLM_v2
      2. Calls train_epoch_teacher instead of train_epoch_v2
      3. Saves extra metadata needed by the KD (knowledge distillation)
         pipeline

    Args:
        fold_idx: Zero-based index of this fold.
        train_indices: Indices (into `dataset`) used for training.
        val_indices: Indices (into `dataset`) used for validation.
        dataset: The full training/validation dataset (or Subset) that
            `train_indices`/`val_indices` index into.
        test_loader: DataLoader for the fixed test set, shared across
            all folds.
        args: Parsed argparse Namespace with all training hyperparameters.
        device: Torch device to train on.
        ss_names: Dict mapping SS class index to a display name.
        run_name: Name used to build this run's output directory.

    Returns:
        dict with keys: 'fold_idx', 'fold_path' (checkpoint path),
        'history' (per-epoch metrics), 'best_val_metrics',
        'test_metrics', 'training_time' (seconds).
    """
    
    fold_start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{args.k_folds}")
    print(f"{'='*60}")
    
    fold_outdir = Path(args.outdir) / f'teacher_{run_name}' / f'fold_{fold_idx + 1}'
    fold_outdir.mkdir(parents=True, exist_ok=True)
    
    # Build loaders
    train_subset = Subset(dataset, train_indices.astype(np.int64).tolist())
    val_subset = Subset(dataset, val_indices.astype(np.int64).tolist())
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize the TEACHER MODEL (not plain ProteinMLM_v2)
    # If dataset is a Subset, get the underlying original dataset
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    
    aa_to_n_clusters = base_dataset.get_aa_to_n_clusters()
    
    model = ProteinMLM_v2(
        n_aa=base_dataset.n_aa - 1,
        aa_to_n_clusters=aa_to_n_clusters,
        n_pos=base_dataset.n_pos,
        n_ss=base_dataset.n_ss,
        d_embed=args.d_embed,
        d_hidden=args.d_hidden,
        n_lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        embed_dropout=args.embed_dropout
    ).to(device)
    
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    scheduler = WarmupCosineScheduler(
        opt, 
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.lr * 0.01
    )
    
    # Training loop with early stopping
    best_val_top5 = 0.0
    patience_counter = 0
    history = {
        'train_loss_aa':    [],
        'val_loss_aa':      [],
        'train_loss_cl':    [],
        'val_loss_cl':      [],
        'train_loss_total': [],   # loss_aa + lambda_cl * loss_cl
        'val_loss_total':   [],
        'val_top1_aa':      [],
        'val_top5_aa':      [],
        'val_top5_joint':   [],
    }

    for epoch in range(args.epochs):
        train_metrics = train_epoch_teacher(
            model, train_loader, opt, aa_to_n_clusters,
            lambda_cl=args.lambda_cl,
            grad_clip=args.grad_clip,
            device=device,
            label_smoothing=args.label_smoothing,
            use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha,
            ss_names=ss_names
        )

        val_metrics = eval_epoch_teacher(
            model, val_loader, aa_to_n_clusters,
            device=device, ss_names=ss_names,
            lambda_cl=args.lambda_cl,
            label_smoothing=args.label_smoothing,
        )

        scheduler.step(epoch)

        # Save metrics into the history
        history['train_loss_aa'].append(train_metrics['loss_aa'])
        history['val_loss_aa'].append(val_metrics['loss_aa'])
        history['train_loss_cl'].append(train_metrics['loss_cl'])
        history['val_loss_cl'].append(val_metrics['loss_cl'])
        history['train_loss_total'].append(train_metrics['loss_total'])
        history['val_loss_total'].append(val_metrics['loss_total'])
        history['val_top1_aa'].append(val_metrics['top1_aa'])
        history['val_top5_aa'].append(val_metrics['top5_aa'])
        history['val_top5_joint'].append(val_metrics.get('top5_joint', 0.0))
        
        # Early stopping based on top5_aa
        if val_metrics['top5_aa'] > best_val_top5:
            best_val_top5 = val_metrics['top5_aa']
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': None,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                # Metadata for generate_teacher_predictions.py
                'model_config': {
                    'n_aa': base_dataset.n_aa - 1,
                    'aa_to_n_clusters': aa_to_n_clusters,
                    'n_pos': base_dataset.n_pos,
                    'n_ss': base_dataset.n_ss,
                    'd_embed': args.d_embed,
                    'd_hidden': args.d_hidden,
                    'n_lstm_layers': args.lstm_layers,
                    'dropout': args.dropout,
                    'embed_dropout': args.embed_dropout
                },
                'training_config': {
                    'run_name': run_name,
                    'lambda_cl': args.lambda_cl,
                    'sasa_threshold': args.sasa_threshold
                },
                'ss_names': ss_names
            }
            torch.save(checkpoint, fold_outdir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d}/{args.epochs} | "
                f"TRAIN loss={train_metrics['loss']:.4f} | "
                f"VAL loss={val_metrics['loss']:.4f}  "
                f"top1={val_metrics['top1_aa']:.4f}  "
                f"top5={val_metrics['top5_aa']:.4f}  "
                f"best_top5={best_val_top5:.4f}"
            )
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final test on the best model
    checkpoint = torch.load(fold_outdir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = eval_epoch_teacher(
        model, test_loader, aa_to_n_clusters,
        device=device, ss_names=ss_names
    )
    
    fold_elapsed = time.time() - fold_start_time
    fold_minutes, fold_seconds = divmod(fold_elapsed, 60)
    
    print(f"\nFold {fold_idx + 1} completed in {int(fold_minutes)} min {fold_seconds:05.2f} s")
    print(f"  Best Val Top-5: {best_val_top5:.4f}")
    print(f"  Test Top-5: {test_metrics['top5_aa']:.4f}")
    
    return {
        'fold_idx': fold_idx,
        'fold_path': str(fold_outdir / 'best_model.pt'),
        'history': history,
        'best_val_metrics': checkpoint['val_metrics'],
        'test_metrics': test_metrics,
        'training_time': fold_elapsed
    }


# ============================================================
# AGGREGATE RESULTS
# ============================================================
def aggregate_results(fold_results):
    """
    Aggregates the results of all folds (including per-SS statistics)
    into mean/std summaries.

    Args:
        fold_results: List of per-fold result dicts, as returned by
            `train_single_fold`.

    Returns:
        dict with keys 'validation', 'test', 'ss_stats'; each maps a
        metric name to {'mean', 'std', 'values'} across folds.
    """
    
    aggregated = {
        'validation': {},
        'test': {},
        'ss_stats': {}
    }
    
    # Standard metrics
    val_metrics = defaultdict(list)
    test_metrics = defaultdict(list)
    
    for result in fold_results:
        for key, value in result['best_val_metrics'].items():
            if key != 'ss_stats':
                val_metrics[key].append(value)
        
        for key, value in result['test_metrics'].items():
            if key != 'ss_stats':
                test_metrics[key].append(value)
    
    for metric, values in val_metrics.items():
        aggregated['validation'][metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': [float(v) for v in values]
        }
    
    for metric, values in test_metrics.items():
        aggregated['test'][metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': [float(v) for v in values]
        }
    
    # SS stats
    ss_stats_all_folds = defaultdict(lambda: defaultdict(list))
    
    for result in fold_results:
        if 'ss_stats' in result['test_metrics']:
            for ss_name, metrics in result['test_metrics']['ss_stats'].items():
                for metric_name, value in metrics.items():
                    ss_stats_all_folds[ss_name][metric_name].append(value)
    
    for ss_name, metrics_dict in ss_stats_all_folds.items():
        aggregated['ss_stats'][ss_name] = {}
        for metric_name, values in metrics_dict.items():
            aggregated['ss_stats'][ss_name][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }
    
    return aggregated


# ============================================================
# PRINT FINAL RESULTS
# ============================================================
def print_final_results(aggregated):
    """Prints the final aggregated results, including per-SS statistics."""
    
    print("\n" + "="*60)
    print("K-FOLD CROSS VALIDATION RESULTS (TEACHER)")
    print("="*60)
    
    print("\nVALIDATION:")
    for metric in ['top1_aa', 'top5_aa', 'top10_aa', 'top20_aa']:
        if metric in aggregated['validation']:
            stats = aggregated['validation'][metric]
            print(f"  {metric:15s}: {stats['mean']:.3f} +/- {stats['std']:.3f} "
                  f"({stats['mean']*100:.1f}% +/- {stats['std']*100:.1f}%)")
    
    print("\nTEST SET:")
    for metric in ['top1_aa', 'top5_aa', 'top10_aa', 'top20_aa', 
                   'top5_joint', 'top10_joint', 'acc_cl_given_true_aa']:
        if metric in aggregated['test']:
            stats = aggregated['test'][metric]
            print(f"  {metric:25s}: {stats['mean']:.3f} +/- {stats['std']:.3f} "
                  f"({stats['mean']*100:.1f}% +/- {stats['std']*100:.1f}%)")
    
    
    # SS stats
    if aggregated.get('ss_stats'):
        print("\n" + "="*60)
        print("SECONDARY STRUCTURE STATISTICS (Test Set)")
        print("="*60)
        
        for ss_name in sorted(aggregated['ss_stats'].keys()):
            ss_dict = aggregated['ss_stats'][ss_name]
            print(f"\n  {ss_name}:")
            
            for metric in ['top1_aa', 'top5_aa', 'top10_aa', 'top5_joint']:
                if metric in ss_dict:
                    stats = ss_dict[metric]
                    print(f"    {metric:15s}: {stats['mean']:.3f} +/- {stats['std']:.3f} "
                          f"({stats['mean']*100:.1f}% +/- {stats['std']*100:.1f}%)")
    
    print("="*60)


# ============================================================
# MAIN K-FOLD CV
# ============================================================
def train_kfold_cv(args):
    """
    Main K-Fold CV routine for the teacher model.

    Two modes:
      A) LEGACY (leave-one-protein-class-out): if --protein_metadata and
         --leave_out_protein_class are provided, train and test are both
         carved out of the same --data (original behavior).
      B) NEW (train/test on separate npz files): if --test_data is
         provided, the WHOLE --data set is used for K-fold train/val,
         and --test_data (a separate npz, e.g. fibrils) is used as a
         fixed test set shared by all folds.

    Args:
        args: Parsed argparse Namespace with all training hyperparameters
            (see the CLI argument definitions in `__main__` below).

    Returns:
        dict: The aggregated results, as returned by `aggregate_results`.
    """

    total_start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    legacy_mode = bool(args.protein_metadata and args.leave_out_protein_class)

    print("="*60)
    print("K-FOLD CROSS VALIDATION TRAINING (TEACHER)")
    if legacy_mode:
        print(f"Mode: LEAVE-ONE-PROTEIN-CLASS-OUT")
        print(f"Leave-out protein class: {args.leave_out_protein_class}")
    else:
        print(f"Mode: TRAIN/VAL (K-fold) on --data, fixed TEST on --test_data")
        print(f"  --data:      {args.data}")
        print(f"  --test_data: {args.test_data}")
    print(f"SASA threshold for exposed: {args.sasa_threshold}")
    print("="*60)

    # Load the rotamer library (shared between train and test)
    rotlib = load_rotamer_library(args.rotamer_csv)

    # Load the training/validation dataset
    ds_full = Windows3Dataset(
        args.data,
        rotamer_lib=rotlib,
        mask_central=True,
        seed=args.seed
    )

    if legacy_mode:
        # ====================================================
        # MODE A: LEAVE-ONE-PROTEIN-CLASS-OUT (original)
        # ====================================================
        with open(args.protein_metadata, 'r') as f:
            protein_metadata = json.load(f)

        protein_classes = {}
        for idx_str, metadata in protein_metadata.items():
            if isinstance(metadata, dict) and 'protein_class' in metadata:
                protein_classes[int(idx_str)] = metadata['protein_class']
            else:
                raise ValueError(f"Metadata format error at index {idx_str}. Expected dict with 'protein_class' key.")

        unique_classes = sorted(set(protein_classes.values()))
        print(f"Unique protein classes in the dataset: {len(unique_classes)}")
        if len(unique_classes) <= 20:
            print(f"Classes: {unique_classes}")
        else:
            print(f"Classes (first 10): {unique_classes[:10]}...")
            print(f"       (last 5): {unique_classes[-5:]}")

        test_indices = []
        train_indices_pool = []

        for idx, prot_class in protein_classes.items():
            if prot_class == args.leave_out_protein_class:
                test_indices.append(idx)
            else:
                train_indices_pool.append(idx)

        if len(test_indices) == 0:
            raise ValueError(f"No windows found for protein class: {args.leave_out_protein_class}")

        print(f"\n[LEAVE-ONE-PROTEIN-CLASS-OUT]")
        print(f"  Target protein class: {args.leave_out_protein_class}")
        print(f"  Test windows: {len(test_indices)}")
        print(f"  Training pool: {len(train_indices_pool)}")

        from torch.utils.data import Subset
        ds_train = Subset(ds_full, train_indices_pool)
        ds_test = Subset(ds_full, test_indices)

        run_name = args.leave_out_protein_class

    else:
        # ====================================================
        # MODE B: TRAIN/VAL K-FOLD on --data, TEST on --test_data
        # ====================================================
        if not args.test_data:
            raise ValueError("You must provide --test_data (or --protein_metadata + "
                              "--leave_out_protein_class for legacy mode).")

        ds_train = ds_full  # the whole --data set, used for K-fold train/val

        ds_test = Windows3Dataset(
            args.test_data,
            rotamer_lib=rotlib,
            mask_central=True,
            seed=args.seed
        )

        print(f"\n[TRAIN/VAL on --data, TEST on --test_data]")
        print(f"  Train/val pool (--data):  {len(ds_train)} windows")
        print(f"  Test pool (--test_data):  {len(ds_test)} windows")

        run_name = args.run_name

    # SS names (generic, used only for printing/labeling)
    base_for_ss = ds_full
    ss_names = getattr(base_for_ss, 'ss_names', {i: f"SS_{i}" for i in range(8)})
    print(f"\nSecondary Structure Types: {ss_names}")

    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create folds: exposed/buried-stratified K-fold over the train/val pool
    # (in both modes, ds_train is already the correct pool:
    #  a filtered Subset in legacy mode, the whole ds_full in mode B)
    folds = create_kfold_splits(
        ds_train,
        k_folds=args.k_folds,
        seed=args.seed,
        validation_mode=args.validation_mode,
        sasa_threshold=args.sasa_threshold,
        protein_metadata=None,
        leave_out_protein=None
    )

    # Train each fold
    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        result = train_single_fold(
            fold_idx, train_indices, val_indices,
            ds_train, test_loader, args, device, ss_names, run_name
        )
        fold_results.append(result)

    # Aggregate results
    aggregated = aggregate_results(fold_results)

    outdir = Path(args.outdir) / f"teacher_{run_name}"
    outdir.mkdir(parents=True, exist_ok=True)

    total_elapsed = time.time() - total_start_time
    total_minutes, total_seconds = divmod(total_elapsed, 60)

    fold_times = [r['training_time'] for r in fold_results]

    with open(outdir / 'kfold_results.json', 'w') as f:
        json.dump({
            'aggregated': aggregated,
            'args': vars(args),
            'n_folds': len(fold_results),
            'sasa_threshold': args.sasa_threshold,
            'run_name': run_name,
            'legacy_leave_one_out': legacy_mode,
            'ss_names': ss_names,
            'training_times': {
                'total_seconds': total_elapsed,
                'total_formatted': f"{int(total_minutes)} min {total_seconds:05.2f} s",
                'per_fold_seconds': fold_times,
                'mean_fold_time': float(np.mean(fold_times)),
                'std_fold_time': float(np.std(fold_times))
            }
        }, f, indent=2)

    # Save the history of each fold
    for fold_idx, result in enumerate(fold_results):
        with open(outdir / f'fold_{fold_idx+1}_history.json', 'w') as f:
            json.dump(result['history'], f, indent=2)

    # Find and copy the best model
    test_top5_scores = [r['test_metrics']['top5_aa'] for r in fold_results]
    best_fold_idx = test_top5_scores.index(max(test_top5_scores))
    best_fold_path = fold_results[best_fold_idx]['fold_path']

    shutil.copy(best_fold_path, outdir / 'best_teacher.pt')

    best_info = {
        'best_fold': best_fold_idx + 1,
        'best_test_top5_aa': test_top5_scores[best_fold_idx],
        'all_folds_test_top5': test_top5_scores,
        'model_path': str(best_fold_path),
        'sasa_threshold': args.sasa_threshold,
        'run_name': run_name,
        'ss_names': ss_names
    }

    with open(outdir / 'best_fold_info.json', 'w') as f:
        json.dump(best_info, f, indent=2)

    print_final_results(aggregated)

    print(f"\n{'='*60}")
    print(f"BEST MODEL: Fold {best_fold_idx + 1} (Test Top-5 = {test_top5_scores[best_fold_idx]:.4f})")
    print(f"Saved to: {outdir / 'best_teacher.pt'}")
    print(f"{'='*60}")

    print(f"\nTOTAL TRAINING TIME:")
    print(f"   {int(total_minutes)} min {total_seconds:05.2f} s (total: {total_elapsed:.1f} s)")
    print(f"   Average per fold: {np.mean(fold_times):.1f} s +/- {np.std(fold_times):.1f} s")
    print("="*60)

    return aggregated


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="K-Fold CV for Teacher Model (with Aggregation Head)")
    
    # Data
    ap.add_argument("--data", required=True, help="Path to the complete training/validation dataset (.npz file).")
    ap.add_argument("--rotamer_csv", default="rotamer_localmax_maxdens.csv",
                    help="Path to the rotamer library CSV used to build/validate cluster assignments (default: rotamer_localmax_maxdens.csv).")
    ap.add_argument("--outdir", default="teachers/",
                    help="Directory where checkpoints, histories, and result files will be written (default: teachers/).")
    
    # Teacher / KD specific arguments (LEGACY: leave-one-protein-out on the same --data)
    ap.add_argument("--protein_metadata", default=None,
                    help="[LEGACY] JSON mapping {window_idx: {pdb_id, protein_class_id, protein_class}}. "
                         "If provided together with --leave_out_protein_class, uses the old "
                         "leave-one-protein-class-out behavior on the same --data.")
    ap.add_argument("--leave_out_protein_class", default=None,
                    help="[LEGACY] Protein class to exclude from training (requires --protein_metadata). "
                         "Examples: 'Tau', 'alpha-synuclein', 'Amyloid-beta(1-42)', 'SOD1', etc.")

    # NEW MODE: K-fold train/val on --data, fixed test on --test_data
    ap.add_argument("--test_data", default=None,
                    help="Separate .npz file to use as a fixed TEST set for every fold "
                         "(e.g. fibril windows). Ignored if using legacy mode.")
    ap.add_argument("--run_name", default="globular_vs_fibrils",
                    help="Name used for the output folder (new mode only).")
    
    # K-Fold
    ap.add_argument("--k_folds", type=int, default=5,
                    help="Number of cross-validation folds (default: 5).")
    ap.add_argument("--validation_mode", type=str, default='all',
                    choices=['all', 'exposed', 'buried'],
                    help="Which residues to include in the validation split: 'all', 'exposed' only, or 'buried' only (default: all).")
    ap.add_argument("--sasa_threshold", type=float, default=0.1199,
                    help="SASA threshold to define exposed residues")
    
    # Training
    ap.add_argument("--epochs", type=int, default=200,
                    help="Maximum number of training epochs per fold (default: 200).")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="Number of windows per training/evaluation batch (default: 32).")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Initial learning rate for the AdamW optimizer (default: 1e-3).")
    ap.add_argument("--weight_decay", type=float, default=1e-3,
                    help="Weight decay (L2 regularization) for the AdamW optimizer (default: 1e-3).")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="Maximum gradient norm for gradient clipping (default: 1.0).")
    ap.add_argument("--patience", type=int, default=35,
                    help="Number of epochs without validation improvement before early stopping (default: 35).")
    
    # Model
    ap.add_argument("--d_embed", type=int, default=256,
                    help="Embedding dimension for amino acid/position/SS features (default: 256).")
    ap.add_argument("--d_hidden", type=int, default=512,
                    help="Hidden dimension of the LSTM/model layers (default: 512).")
    ap.add_argument("--lstm_layers", type=int, default=3,
                    help="Number of stacked LSTM layers in the model (default: 3).")
    ap.add_argument("--dropout", type=float, default=0.25,
                    help="Dropout probability applied within the model (default: 0.25).")
    ap.add_argument("--lambda_cl", type=float, default=1.0,
                    help="Weight for cluster loss")
    
    # Anti-overfitting
    ap.add_argument("--embed_dropout", type=float, default=0.3,
                    help="Dropout probability applied to the input embeddings (default: 0.3).")
    ap.add_argument("--label_smoothing", type=float, default=0.08,
                    help="Label smoothing factor applied to the classification losses (default: 0.08).")
    ap.add_argument("--use_mixup", action="store_true",
                    help="Enable mixup data augmentation during training.")
    ap.add_argument("--mixup_alpha", type=float, default=0.2,
                    help="Alpha parameter of the mixup Beta distribution, used only if --use_mixup is set (default: 0.2).")
    ap.add_argument("--warmup_epochs", type=int, default=5,
                    help="Number of learning-rate warmup epochs before the cosine schedule begins (default: 5).")
    
    # Other
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed used for reproducibility (data splits, model init, mixup, etc.) (default: 42).")
    ap.add_argument("--cpu", action="store_true",
                    help="Force training on CPU even if a CUDA GPU is available.")
    
    args = ap.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train_kfold_cv(args)
