"""
PLOT TRAINING CURVES - Teacher & Student
==========================================

Analizza i file history.json salvati durante il training K-Fold
e genera grafici diagnostici per identificare overfitting.

Funzionalità:
  1. Plot training vs validation loss per ogni fold
  2. Plot medie aggregate con confidence intervals
  3. Diagnostica overfitting (gap train-val, plateau, oscillazioni)
  4. Confronto teacher vs student
  5. Analisi delle singole componenti di loss

Uso:
    python plot_training_curves.py --model student --results_dir runs/student_lopo_tau/
            python plot_training_curves.py --model teacher --results_dir teachers1/teacher_α-synuclein
"""
"""
PLOT TRAINING CURVES - Teacher & Student
==========================================

Analizza i file history.json salvati durante il training K-Fold
e genera grafici diagnostici per identificare overfitting.

Funzionalità:
  1. Plot training vs validation loss per ogni fold
  2. Plot medie aggregate con confidence intervals
  3. Diagnostica overfitting (gap train-val, plateau, oscillazioni)
  4. Confronto teacher vs student
  5. Analisi delle singole componenti di loss

Uso:
    python plot_training_curves.py --model student --results_dir runs/student_lopo_tau/
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Stile professionale
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_fold_histories(results_dir: Path, model_type: str = 'teacher') -> Dict[int, Dict]:
    """
    Carica tutti i file history.json dai fold
    
    Args:
        results_dir: Directory con i risultati K-Fold
        model_type: 'teacher' o 'student'
    
    Returns:
        Dict[fold_idx, history_dict]
    """
    results_dir = Path(results_dir)
    histories = {}
    
    if model_type == 'teacher':
        # Nuovo formato: fold_1/history.json  (teacher_kfold_train_complete.py aggiornato)
        for fold_dir in sorted(results_dir.glob('fold_*')):
            if fold_dir.is_dir():
                history_file = fold_dir / 'history.json'
                if history_file.exists():
                    fold_idx = int(fold_dir.name.split('_')[1])
                    with open(history_file) as f:
                        histories[fold_idx] = json.load(f)
        # Vecchio formato (retrocompatibilità): fold_1_history.json
        if not histories:
            for fold_file in sorted(results_dir.glob('fold_*_history.json')):
                fold_idx = int(fold_file.stem.split('_')[1])
                with open(fold_file) as f:
                    histories[fold_idx] = json.load(f)
    
    elif model_type == 'student':
        # Student: fold_1/history.json, fold_2/history.json, ...
        for fold_dir in sorted(results_dir.glob('fold_*')):
            if fold_dir.is_dir():
                history_file = fold_dir / 'history.json'
                if history_file.exists():
                    fold_idx = int(fold_dir.name.split('_')[1])
                    with open(history_file) as f:
                        histories[fold_idx] = json.load(f)
    
    print(f"Loaded {len(histories)} fold histories from {results_dir}")
    return histories


def diagnose_overfitting(history: Dict, metric: str = 'loss') -> Dict:
    """
    Diagnostica overfitting da una singola history
    
    Args:
        history: dict con train_metric e val_metric
        metric: nome della metrica (es. 'loss', 'loss_aa', 'total')
    
    Returns:
        Dict con diagnostica:
        - final_gap: gap finale tra train e val
        - max_gap: gap massimo
        - gap_trend: 'increasing', 'stable', 'decreasing'
        - val_plateau: True se val loss in plateau
        - oscillations: std delle ultime 20 epochs
        - overfitting_score: 0-1 (0=no overfit, 1=severe overfit)
    """
    train_key = f'train_{metric}'
    val_key = f'val_{metric}'
    
    if train_key not in history or val_key not in history:
        return {'error': f'Metric {metric} not found in history'}
    
    train_loss = np.array(history[train_key])
    val_loss = np.array(history[val_key])
    
    # Gap train-val
    gap = val_loss - train_loss
    final_gap = gap[-1]
    max_gap = np.max(gap)
    
    # Trend del gap (ultimi 30% epochs)
    window = max(10, len(gap) // 3)
    gap_early = np.mean(gap[:window])
    gap_late = np.mean(gap[-window:])
    
    if gap_late > gap_early * 1.2:
        gap_trend = 'increasing'  # ⚠️ Segnale di overfitting
    elif gap_late < gap_early * 0.8:
        gap_trend = 'decreasing'
    else:
        gap_trend = 'stable'
    
    # Plateau nella validation loss
    val_window = val_loss[-20:] if len(val_loss) >= 20 else val_loss[-10:]
    val_std = np.std(val_window)
    val_mean = np.mean(val_window)
    val_plateau = (val_std / (val_mean + 1e-8)) < 0.02  # < 2% variazione
    
    # Oscillazioni
    oscillations = np.std(val_window)
    
    # Overfitting score (euristico)
    # - gap_score: quanto è grande il gap rispetto a val_loss
    # - trend_score: penalizza gap crescente
    # - plateau_score: validation in plateau ma train scende = overfit
    
    gap_score = min(1.0, final_gap / (np.mean(val_loss) + 1e-8))
    trend_score = 0.5 if gap_trend == 'increasing' else 0.0
    lookback = min(20, len(train_loss) - 1)
    plateau_score = 0.3 if val_plateau and train_loss[-1] < train_loss[-lookback] else 0.0


    overfitting_score = np.clip(gap_score + trend_score + plateau_score, 0, 1)
    
    return {
        'final_gap': float(final_gap),
        'max_gap': float(max_gap),
        'gap_trend': gap_trend,
        'val_plateau': bool(val_plateau),
        'oscillations': float(oscillations),
        'overfitting_score': float(overfitting_score),
        'final_train_loss': float(train_loss[-1]),
        'final_val_loss': float(val_loss[-1])
    }


def plot_single_fold(history: Dict, fold_idx: int, model_type: str, 
                     save_path: Path = None, metrics: List[str] = None):
    """
    Plot training curves per un singolo fold
    
    Args:
        history: dict con metriche
        fold_idx: indice del fold
        model_type: 'teacher' o 'student'
        save_path: dove salvare il plot
        metrics: lista di metriche da plottare
    """
    if metrics is None:
        if model_type == 'teacher':
            metrics = ['loss_aa', 'loss_cl', 'loss_total']
        else:
            metrics = ['total', 'ce', 'kl', 'emb', 'agg']
    
    # Per il teacher aggiungi anche accuracy e pearson se disponibili
    extra_teacher_panels = []
    if model_type == 'teacher':
        if 'val_top5_aa' in history:
            extra_teacher_panels.append('accuracy')
        if 'val_agg_pearson' in history:
            extra_teacher_panels.append('agg_diagnostics')
    
    n_metrics = len(metrics)
    # Conta i panel extra per il teacher (accuracy + agg_diagnostics)
    n_extra = len(extra_teacher_panels) if model_type == 'teacher' else 0
    n_total = n_metrics + n_extra
    n_cols = max(2, (n_total + 1) // 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 8))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history and val_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            
            ax.plot(epochs, history[train_key], 
                   label=f'Train {metric}', linewidth=2, alpha=0.8)
            ax.plot(epochs, history[val_key], 
                   label=f'Val {metric}', linewidth=2, alpha=0.8)
            
            # Diagnostica
            diag = diagnose_overfitting(history, metric)
            ovf_score = diag['overfitting_score']
            
            # Colore titolo basato su overfitting score
            if ovf_score > 0.7:
                title_color = 'red'
                status = '⚠️ HIGH OVERFIT'
            elif ovf_score > 0.4:
                title_color = 'orange'
                status = '⚠️ Moderate Overfit'
            else:
                title_color = 'green'
                status = '✓ Healthy'
            
            ax.set_title(f'{metric.upper()} - {status}\nGap: {diag["final_gap"]:.4f}',
                        color=title_color, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{metric}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Panel extra per teacher: accuracy top-1/top-5 e AGG diagnostics
    extra_idx = n_metrics
    if model_type == 'teacher':
        if 'accuracy' in extra_teacher_panels and extra_idx < len(axes):
            ax = axes[extra_idx]
            epochs_acc = range(1, len(history['val_top1_aa']) + 1)
            ax.plot(epochs_acc, history['val_top1_aa'],  label='Val Top-1',   linewidth=2, color='#E63946')
            ax.plot(epochs_acc, history['val_top5_aa'],  label='Val Top-5',   linewidth=2, color='#2A9D8F')
            if 'val_top5_joint' in history:
                ax.plot(epochs_acc, history['val_top5_joint'], label='Val Top-5 Joint', linewidth=2, color='#E9C46A', linestyle='--')
            ax.set_title(f'ACCURACY\nTop-5: {history["val_top5_aa"][-1]:.3f}', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            extra_idx += 1

        if 'agg_diagnostics' in extra_teacher_panels and extra_idx < len(axes):
            ax = axes[extra_idx]
            epochs_p = range(1, len(history['val_agg_pearson']) + 1)
            ax2 = ax.twinx()
            ax.plot(epochs_p,  history['val_agg_pearson'], label='Pearson r', linewidth=2, color='#6A0572')
            ax2.plot(epochs_p, history['val_agg_mae'],     label='MAE',       linewidth=2, color='#F4A261', linestyle='--')
            ax.set_title(f'AGG DIAGNOSTICS\nPearson: {history["val_agg_pearson"][-1]:.3f}', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Pearson r', color='#6A0572')
            ax2.set_ylabel('MAE', color='#F4A261')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
            ax.grid(True, alpha=0.3)
            extra_idx += 1

    # Rimuovi subplot vuoti
    for idx in range(extra_idx, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle(f'{model_type.upper()} - Fold {fold_idx} Training Curves', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_aggregate_curves(histories: Dict, model_type: str, 
                          save_path: Path = None, metric: str = 'loss',
                          ylim: tuple = None):
    """
    Plot curve aggregate di tutti i fold con confidence intervals
    
    Args:
        histories: Dict[fold_idx, history]
        model_type: 'teacher' o 'student'
        save_path: dove salvare
        metric: metrica da plottare (es. 'loss', 'total')
        ylim: tupla (ymin, ymax) per impostare i limiti dell'asse Y, es. (0, 1)
    """
    train_key = f'train_{metric}'
    val_key = f'val_{metric}'
    
    # Colleziona tutte le curve
    all_train = []
    all_val = []
    max_epochs = 0
    
    for fold_idx, history in histories.items():
        if train_key in history and val_key in history:
            all_train.append(history[train_key])
            all_val.append(history[val_key])
            max_epochs = max(max_epochs, len(history[train_key]))
    
    if not all_train:
        print(f"No data for metric: {metric}")
        return None
    
    # Padding a max_epochs (con NaN)
    all_train_padded = []
    all_val_padded = []
    
    for train, val in zip(all_train, all_val):
        train_pad = list(train) + [np.nan] * (max_epochs - len(train))
        val_pad = list(val) + [np.nan] * (max_epochs - len(val))
        all_train_padded.append(train_pad)
        all_val_padded.append(val_pad)
    
    all_train_padded = np.array(all_train_padded)
    all_val_padded = np.array(all_val_padded)
    
    # Media e std (ignorando NaN)
    train_mean = np.nanmean(all_train_padded, axis=0)
    train_std = np.nanstd(all_train_padded, axis=0)
    val_mean = np.nanmean(all_val_padded, axis=0)
    val_std = np.nanstd(all_val_padded, axis=0)
    
    epochs = np.arange(1, max_epochs + 1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Train
    ax.plot(epochs, train_mean, label='Train (mean)', 
           linewidth=4, color='#2E86AB', alpha=0.9)
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='#2E86AB', label='Train (±1 std)')
    
    # Val
    ax.plot(epochs, val_mean, label='Validation (mean)', 
           linewidth=4, color='#A23B72', alpha=0.9)
    ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='#A23B72', label='Val (±1 std)')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(f'Loss Curve\nMean ± Std across {len(histories)} folds',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, frameon=True)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved: {save_path}")

        # Multi-format export solo per loss_aa
        if metric == 'loss_aa':
            base = Path(save_path).with_suffix('')
            plt.savefig(base.with_suffix('.svg'),  format='svg',  bbox_inches='tight')
            plt.savefig(base.with_suffix('.pdf'),  format='pdf',  bbox_inches='tight')
            plt.savefig(base.with_suffix('.tiff'), format='tiff', dpi=600,
                        bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
            print(f"Saved multi-format (svg/pdf/tiff/png) for loss_aa")

    return fig


def compare_teacher_student(teacher_histories: Dict, student_histories: Dict,
                            save_path: Path = None):
    """
    Confronta le curve di teacher e student affiancate
    
    Args:
        teacher_histories: Dict[fold_idx, history] del teacher
        student_histories: Dict[fold_idx, history] dello student
        save_path: dove salvare
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Teacher
    ax = axes[0]
    teacher_metric = 'loss'
    train_key = f'train_{teacher_metric}'
    val_key = f'val_{teacher_metric}'
    
    all_train = []
    all_val = []
    for history in teacher_histories.values():
        if train_key in history and val_key in history:
            all_train.append(history[train_key])
            all_val.append(history[val_key])
    
    if all_train:
        max_len = max(len(t) for t in all_train)
        all_train_pad = [list(t) + [np.nan]*(max_len-len(t)) for t in all_train]
        all_val_pad = [list(v) + [np.nan]*(max_len-len(v)) for v in all_val]
        
        train_mean = np.nanmean(all_train_pad, axis=0)
        val_mean = np.nanmean(all_val_pad, axis=0)
        epochs = np.arange(1, len(train_mean) + 1)
        
        ax.plot(epochs, train_mean, label='Train', linewidth=2, color='#2E86AB')
        ax.plot(epochs, val_mean, label='Validation', linewidth=2, color='#A23B72')
        ax.set_title('TEACHER', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Student
    ax = axes[1]
    student_metric = 'total'
    train_key = f'train_{student_metric}'
    val_key = f'val_{student_metric}'
    
    all_train = []
    all_val = []
    for history in student_histories.values():
        if train_key in history and val_key in history:
            all_train.append(history[train_key])
            all_val.append(history[val_key])
    
    if all_train:
        max_len = max(len(t) for t in all_train)
        all_train_pad = [list(t) + [np.nan]*(max_len-len(t)) for t in all_train]
        all_val_pad = [list(v) + [np.nan]*(max_len-len(v)) for v in all_val]
        
        train_mean = np.nanmean(all_train_pad, axis=0)
        val_mean = np.nanmean(all_val_pad, axis=0)
        epochs = np.arange(1, len(train_mean) + 1)
        
        ax.plot(epochs, train_mean, label='Train', linewidth=2, color='#2E86AB')
        ax.plot(epochs, val_mean, label='Validation', linewidth=2, color='#A23B72')
        ax.set_title('STUDENT', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Teacher vs Student - Training Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_overfitting_report(histories: Dict, model_type: str, 
                                save_path: Path = None) -> str:
    """
    Genera report testuale sull'overfitting
    
    Args:
        histories: Dict[fold_idx, history]
        model_type: 'teacher' o 'student'
        save_path: file .txt dove salvare
    
    Returns:
        report string
    """
    if model_type == 'teacher':
        metrics = ['loss_aa', 'loss_cl', 'loss_total']
    else:
        metrics = ['total', 'ce', 'kl', 'emb', 'agg']
    
    report = []
    report.append("=" * 70)
    report.append(f"OVERFITTING DIAGNOSTIC REPORT - {model_type.upper()}")
    report.append("=" * 70)
    report.append(f"\nAnalyzing {len(histories)} folds\n")
    
    for metric in metrics:
        report.append(f"\n{'='*70}")
        report.append(f"METRIC: {metric.upper()}")
        report.append(f"{'='*70}")
        
        all_diagnostics = []
        for fold_idx, history in histories.items():
            diag = diagnose_overfitting(history, metric)
            if 'error' not in diag:
                all_diagnostics.append((fold_idx, diag))
        
        if not all_diagnostics:
            report.append(f"  No data available for {metric}")
            continue
        
        # Statistiche aggregate
        ovf_scores = [d[1]['overfitting_score'] for d in all_diagnostics]
        gaps = [d[1]['final_gap'] for d in all_diagnostics]
        
        mean_ovf = np.mean(ovf_scores)
        mean_gap = np.mean(gaps)
        
        # Severity
        if mean_ovf > 0.7:
            severity = "🔴 SEVERE OVERFITTING"
        elif mean_ovf > 0.4:
            severity = "🟡 MODERATE OVERFITTING"
        else:
            severity = "🟢 HEALTHY (No significant overfitting)"
        
        report.append(f"\n  Overall Assessment: {severity}")
        report.append(f"  Mean Overfitting Score: {mean_ovf:.3f} (0=healthy, 1=severe)")
        report.append(f"  Mean Train-Val Gap: {mean_gap:.4f}")
        
        report.append(f"\n  Per-Fold Details:")
        report.append(f"  {'Fold':<6} {'Ovf.Score':<12} {'Final Gap':<12} {'Trend':<12} {'Val Plateau':<12}")
        report.append(f"  {'-'*60}")
        
        for fold_idx, diag in all_diagnostics:
            report.append(
                f"  {fold_idx:<6} {diag['overfitting_score']:<12.3f} "
                f"{diag['final_gap']:<12.4f} {diag['gap_trend']:<12} "
                f"{'Yes' if diag['val_plateau'] else 'No':<12}"
            )
        
        # Raccomandazioni
        report.append(f"\n  📌 Recommendations:")
        if mean_ovf > 0.7:
            report.append("     - ⚠️ Critical: Model is severely overfitting!")
            report.append("     - Increase regularization (dropout, weight_decay)")
            report.append("     - Reduce model capacity (fewer layers/hidden units)")
            report.append("     - Use more data augmentation")
            report.append("     - Consider early stopping earlier")
        elif mean_ovf > 0.4:
            report.append("     - ⚠️ Warning: Moderate overfitting detected")
            report.append("     - Consider increasing dropout slightly")
            report.append("     - Monitor validation loss more closely")
            report.append("     - Early stopping might help")
        else:
            report.append("     - ✓ Model is training healthily")
            report.append("     - Current regularization seems appropriate")
            report.append("     - Continue with current setup")
    
    report.append(f"\n{'='*70}\n")
    
    report_str = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)
        print(f"Report saved: {save_path}")
    
    return report_str


def _plot_teacher_accuracy_aggregate(histories: Dict, output_dir: Path):
    """Plot aggregato accuracy top-1, top-5, top-5-joint del teacher."""
    first = next(iter(histories.values()))
    if 'val_top5_aa' not in first:
        print("  Skipping accuracy plot: val_top5_aa not in history")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'val_top1_aa': ('#E63946', 'Top-1 AA'),
              'val_top5_aa': ('#2A9D8F', 'Top-5 AA'),
              'val_top5_joint': ('#E9C46A', 'Top-5 Joint')}

    for key, (color, label) in colors.items():
        all_curves = []
        for history in histories.values():
            if key in history:
                all_curves.append(history[key])
        if not all_curves:
            continue
        max_len = max(len(c) for c in all_curves)
        padded = np.array([c + [np.nan] * (max_len - len(c)) for c in all_curves])
        mean = np.nanmean(padded, axis=0)
        std  = np.nanstd(padded,  axis=0)
        epochs = np.arange(1, max_len + 1)
        ax.plot(epochs, mean, label=f'{label} (mean)', linewidth=2.5, color=color)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('TEACHER - Aggregate Accuracy\nMean ± Std across folds',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = output_dir / 'teacher_aggregate_accuracy.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def _plot_teacher_agg_diagnostics_aggregate(histories: Dict, output_dir: Path):
    """Plot aggregato Pearson r e MAE dell'aggregation head del teacher."""
    first = next(iter(histories.values()))
    if 'val_agg_pearson' not in first:
        print("  Skipping agg diagnostics plot: val_agg_pearson not in history")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for key, ax, color, label, ls in [
        ('val_agg_pearson', ax1, '#6A0572', 'Pearson r (mean)', '-'),
        ('val_agg_mae',     ax2, '#F4A261', 'MAE (mean)',        '--'),
    ]:
        all_curves = []
        for history in histories.values():
            if key in history:
                all_curves.append(history[key])
        if not all_curves:
            continue
        max_len = max(len(c) for c in all_curves)
        padded = np.array([c + [np.nan] * (max_len - len(c)) for c in all_curves])
        mean = np.nanmean(padded, axis=0)
        std  = np.nanstd(padded,  axis=0)
        epochs = np.arange(1, max_len + 1)
        ax.plot(epochs, mean, label=label, linewidth=2.5, color=color, linestyle=ls)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Pearson r', fontsize=12, color='#6A0572')
    ax2.set_ylabel('MAE', fontsize=12, color='#F4A261')
    ax1.set_title('TEACHER - Aggregation Head Diagnostics\nMean ± Std across folds',
                  fontsize=13, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = output_dir / 'teacher_aggregate_agg_diagnostics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training curves and diagnose overfitting')
    
    # Mode selection
    parser.add_argument('--model', type=str, choices=['teacher', 'student'],
                       help='Which model to analyze')
    parser.add_argument('--results_dir', type=str,
                       help='Directory with K-Fold results')
    parser.add_argument('--compare', action='store_true',
                       help='Compare teacher vs student')
    parser.add_argument('--teacher_dir', type=str,
                       help='Teacher results directory (for comparison)')
    parser.add_argument('--student_dir', type=str,
                       help='Student results directory (for comparison)')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Where to save plots')
    parser.add_argument('--ylim', type=float, nargs=2, metavar=('YMIN', 'YMAX'),
                       help='Limiti asse Y per i plot aggregati, es. --ylim 0 1')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.compare:
        # Confronto teacher vs student
        if not args.teacher_dir or not args.student_dir:
            parser.error("--compare requires --teacher_dir and --student_dir")
        
        print("\n" + "="*70)
        print("COMPARING TEACHER vs STUDENT")
        print("="*70 + "\n")
        
        teacher_histories = load_fold_histories(Path(args.teacher_dir), 'teacher')
        student_histories = load_fold_histories(Path(args.student_dir), 'student')
        
        # Plot confronto
        compare_teacher_student(
            teacher_histories, 
            student_histories,
            save_path=output_dir / 'comparison_teacher_vs_student.png'
        )
        
        # Report separati
        print("\n--- TEACHER REPORT ---")
        teacher_report = generate_overfitting_report(
            teacher_histories, 
            'teacher',
            save_path=output_dir / 'teacher_overfitting_report.txt'
        )
        print(teacher_report)
        
        print("\n--- STUDENT REPORT ---")
        student_report = generate_overfitting_report(
            student_histories,
            'student', 
            save_path=output_dir / 'student_overfitting_report.txt'
        )
        print(student_report)
    
    else:
        # Analisi singolo modello
        if not args.model or not args.results_dir:
            parser.error("Single model analysis requires --model and --results_dir")
        
        print(f"\nAnalyzing {args.model.upper()} model...")
        print("="*70 + "\n")
        
        histories = load_fold_histories(Path(args.results_dir), args.model)
        
        if not histories:
            print("ERROR: No history files found!")
            return
        
        # Plot per ogni fold
        print("\nGenerating per-fold plots...")
        for fold_idx, history in histories.items():
            save_path = output_dir / f'{args.model}_fold_{fold_idx}_curves.png'
            plot_single_fold(history, fold_idx, args.model, save_path=save_path)
        
        # Plot aggregato
        print("\nGenerating aggregate plots...")
        if args.model == 'teacher':
            metrics = ['loss_aa', 'loss_cl', 'loss_total']
        else:
            metrics = ['total', 'ce', 'kl', 'emb', 'agg']
        
        for metric in metrics:
            save_path = output_dir / f'{args.model}_aggregate_{metric}.png'
            plot_aggregate_curves(histories, args.model, save_path=save_path, metric=metric,
                                  ylim=tuple(args.ylim) if args.ylim else None)
        
        # Plot accuracy aggregata (teacher)
        if args.model == 'teacher':
            _plot_teacher_accuracy_aggregate(histories, output_dir)
            _plot_teacher_agg_diagnostics_aggregate(histories, output_dir)
        
        # Report diagnostico
        print("\nGenerating overfitting diagnostic report...")
        report = generate_overfitting_report(
            histories, 
            args.model,
            save_path=output_dir / f'{args.model}_overfitting_report.txt'
        )
        print("\n" + report)
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()