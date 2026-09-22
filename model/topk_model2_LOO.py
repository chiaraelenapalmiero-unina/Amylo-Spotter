"""
MLM v2 model with advanced anti-overfitting techniques:
- Label Smoothing
- Embedding Dropout
- Mixup (optional)
- Warmup + Cosine Annealing
- Joint (AA+Cluster) metrics, as in topk_model.py
python topk_model2.py   --data ./windows_all_local.npz   --test_data ./windows_exposed_testset_local.npz   --rotamer_csv rotamer_localmax_maxdens.csv   --outdir runs/mlm_bilstm_final   --epochs 150   --batch_size 32   --val_size 146   --lr 1e-3   --weight_decay 1e-3   --grad_clip 1.0   --d_embed 128   --d_hidden 256   --lstm_layers 2   --dropout 0.3   --lambda_cl 1.0   --patience (20 for buried, 15 for exposed)   --embed_dropout 0.4   --label_smoothing 0.1   --use_mixup   --mixup_alpha 0.2

Adds separate metrics for each secondary structure (SS) type (Helix,
Sheet, Coil, etc.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import math
import time
from collections import defaultdict


# ============================================================
# MODEL v2: BiLSTM + Anti-Overfitting (unchanged architecture)
# ============================================================
class ProteinMLM_v2(nn.Module):
    """
    Masked-language model for protein residue windows: predicts the
    central residue's amino acid identity and rotamer cluster from a
    5-residue window, using amino acid, rotamer cluster, SASA, secondary
    structure (SS), and position embeddings fed through a bidirectional
    LSTM with attention pooling.
    """
    def __init__(self, n_aa, aa_to_n_clusters, n_pos=3, n_ss=8,
                 d_embed=64, d_hidden=64, n_lstm_layers=1, dropout=0.5,
                 embed_dropout=0.3, use_blosum=True):
        """
        Args:
            n_aa: Number of amino acid classes (excluding the MASK
                token; the MASK embedding is added internally as index
                n_aa).
            aa_to_n_clusters: Dict {aa_idx: n_clusters}, from the
                rotamer library, giving the number of valid rotamer
                clusters for each amino acid.
            n_pos: Number of distinct position-in-window values
                (default: 3).
            n_ss: Number of secondary structure classes (default: 8).
            d_embed: Embedding dimension for each input feature
                (amino acid, cluster, SASA, SS, position).
            d_hidden: Hidden size of each LSTM direction (the BiLSTM
                output has size 2 * d_hidden).
            n_lstm_layers: Number of stacked LSTM layers.
            dropout: Dropout probability used in the LSTM (when
                n_lstm_layers > 1), the attention-pooled representation,
                and the classification heads.
            embed_dropout: Dropout probability applied to each input
                embedding before concatenation.
            use_blosum: If True (default), concatenate a fixed BLOSUM62
                row (looked up by amino acid) to the input features as
                an additional evolutionary-similarity signal.
        """
        super().__init__()
        self.n_aa = n_aa
        self.aa_to_n_clusters = aa_to_n_clusters
        self.max_clusters = max(aa_to_n_clusters.values())
        self.use_blosum = use_blosum
        
        if use_blosum:
            self.register_buffer('blosum62', self._init_blosum62())
            blosum_dim = 20
        else:
            blosum_dim = 0
        
        self.emb_aa = nn.Embedding(n_aa + 1, d_embed)
        self.emb_cl = nn.Embedding(self.max_clusters + 1, d_embed)
        self.sasa_projection = nn.Linear(1, d_embed)
        self.emb_ss = nn.Embedding(n_ss, d_embed)
        self.emb_pos = nn.Embedding(n_pos, d_embed)
        
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        lstm_input_size = d_embed * 5 + blosum_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=d_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0
        )
        
        self.layer_norm = nn.LayerNorm(d_hidden * 2)
        
        self.attention = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1)
        )
        
        d_out = d_hidden * 2
        self.dropout = nn.Dropout(dropout)
        
        self.head_aa = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, n_aa)
        )
        
        # VECTORIZED heads_cl: instead of an nn.ModuleDict with 20 separate
        # networks (called one at a time in a Python loop inside forward,
        # with 20 tiny GPU kernel launches per batch), we pack the weights
        # of every head into a single pair of tensors and compute them all
        # with one batched operation (einsum). Same identical architecture
        # per amino acid (Linear -> ReLU -> Dropout -> Linear), same
        # identical per-head Xavier initialization, same identical numeric
        # output -- verified with a direct equivalence test against the
        # ModuleDict version.
        cl_hidden = max(64, d_out // 4)  # e.g. 512*2//4 = 256
        self.cl_hidden = cl_hidden

        W1 = torch.empty(self.n_aa, d_out, cl_hidden)
        b1 = torch.zeros(self.n_aa, cl_hidden)
        W2 = torch.zeros(self.n_aa, cl_hidden, self.max_clusters)
        b2 = torch.zeros(self.n_aa, self.max_clusters)

        valid_mask = torch.zeros(self.n_aa, self.max_clusters, dtype=torch.bool)
        for aa_idx in range(self.n_aa):
            n_cl = aa_to_n_clusters.get(aa_idx, 0)
            # Init identical to nn.Linear + xavier_uniform_(gain=0.8) from
            # the original version, applied independently for each head
            # (exactly as it was with 20 separate nn.Linear modules).
            nn.init.xavier_uniform_(W1[aa_idx], gain=0.8)
            if n_cl > 0:
                nn.init.xavier_uniform_(W2[aa_idx, :, :n_cl], gain=0.8)
                valid_mask[aa_idx, :n_cl] = True

        self.W1_cl = nn.Parameter(W1)
        self.b1_cl = nn.Parameter(b1)
        self.W2_cl = nn.Parameter(W2)
        self.b2_cl = nn.Parameter(b2)
        self.dropout_cl = nn.Dropout(dropout)
        self.register_buffer('valid_mask_cl', valid_mask)
        
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[PROTEIN MLM v2 - SASA + SS] Parameters: {n_params:,}")
        print(f"  Embed dropout: {embed_dropout}, Main dropout: {dropout}")
        print(f"  Using SASA + Secondary Structure (n_ss={n_ss})")
    
    def _init_blosum62(self):
        """Builds the BLOSUM62 substitution matrix, reordered to match AA_ORDER."""
        blosum_standard_order = "ARNDCQEGHILKMFPSTWYV"
        blosum62_standard = np.array([
            [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
            [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
            [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
            [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
            [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
            [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
            [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
            [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
            [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
            [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
            [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
            [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
            [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
            [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
            [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
            [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
            [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
            [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
            [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
            [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],
        ], dtype=np.float32)
        
        new_order = "ACDEFGHIKLMNPQRSTVWY"
        idx_standard = {aa: i for i, aa in enumerate(blosum_standard_order)}
        
        blosum_reordered = np.zeros_like(blosum62_standard)
        for i, aa_i in enumerate(new_order):
            for j, aa_j in enumerate(new_order):
                blosum_reordered[i, j] = blosum62_standard[idx_standard[aa_i], idx_standard[aa_j]]
        
        return torch.tensor(blosum_reordered, dtype=torch.float32)
    
    def _init_weights(self):
        """Initializes Linear, Embedding, and LSTM weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.8)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, batch, mixup_lambda=None, mixup_indices=None):
        """
        Runs a forward pass over a batch of windows.

        Args:
            batch: dict with tensors 'aa', 'cl', 'sasa', 'ss', 'pos',
                each of shape [B, L] (see `Windows3Dataset.__getitem__`).
            mixup_lambda: Optional float mixup interpolation coefficient.
                If provided together with `mixup_indices`, the input
                embeddings are linearly interpolated between each sample
                and the one at `mixup_indices`.
            mixup_indices: Optional LongTensor of shape [B] giving, for
                each sample, the index of the other sample to mix with.

        Returns:
            tuple:
                logits_aa: [B, n_aa] amino acid classification logits.
                logits_cl_all: [B, n_aa, max_clusters] rotamer cluster
                    logits for every (amino acid, cluster) slot, with
                    invalid slots set to -inf via `valid_mask`.
                valid_mask: [n_aa, max_clusters] boolean mask marking
                    which (amino acid, cluster) pairs are valid.
        """
        aa = batch['aa'].long()
        cl = batch['cl'].long()
        sasa = batch['sasa'].float()
        ss = batch['ss'].long()
        pos = batch['pos'].long()
        
        aa_emb = self.embed_dropout(self.emb_aa(aa))
        cl_emb = self.embed_dropout(self.emb_cl(cl))
        sasa_emb = self.sasa_projection(sasa.unsqueeze(-1))
        sasa_emb = self.embed_dropout(sasa_emb)
        ss_emb = self.embed_dropout(self.emb_ss(ss))
        pos_emb = self.emb_pos(pos)
        
        if self.use_blosum:
            aa_clamped = torch.clamp(aa, 0, 19)
            blosum_feat = self.blosum62[aa_clamped]
            x = torch.cat([aa_emb, blosum_feat, cl_emb, sasa_emb, ss_emb, pos_emb], dim=-1)
        else:
            x = torch.cat([aa_emb, cl_emb, sasa_emb, ss_emb, pos_emb], dim=-1)
        
        if mixup_lambda is not None and mixup_indices is not None:
            x = mixup_lambda * x + (1 - mixup_lambda) * x[mixup_indices]
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        attn_scores = self.attention(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        h = (lstm_out * attn_weights).sum(dim=1)
        
        L = aa.size(1)
        center = L // 2
        h_center = lstm_out[:, center, :]
        
        h = (h + h_center) / 2.0
        h = self.dropout(h)
        
        logits_aa = self.head_aa(h)
        
        B = h.size(0)
        device = h.device
        
        # Vectorized version: all 20 heads computed in a single batched
        # operation (2 einsum calls) instead of 20 sequential calls.
        # Same output as: for each aa_idx, heads_cl[aa_idx](h) -> [B, n_cl],
        # packed into [B, n_aa, max_clusters] with the same -inf padding
        # as before for invalid positions.
        hidden_cl = torch.einsum('bd,ndh->bnh', h, self.W1_cl) + self.b1_cl  # [B, n_aa, cl_hidden]
        hidden_cl = F.relu(hidden_cl)
        hidden_cl = self.dropout_cl(hidden_cl)
        logits_cl_all = torch.einsum('bnh,nhc->bnc', hidden_cl, self.W2_cl) + self.b2_cl  # [B, n_aa, max_clusters]

        valid_mask = self.valid_mask_cl
        logits_cl_all = logits_cl_all.masked_fill(~valid_mask.unsqueeze(0), float('-inf'))

        return logits_aa, logits_cl_all, valid_mask


# ============================================================
# COSINE ANNEALING WITH WARMUP (unchanged)
# ============================================================
class WarmupCosineScheduler:
    """
    Learning-rate scheduler combining a linear warmup phase with cosine
    annealing afterward.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        """
        Args:
            optimizer: The torch optimizer whose param group learning
                rates will be updated.
            warmup_epochs: Number of epochs over which the learning rate
                ramps up linearly from 0 to its base value.
            total_epochs: Total number of training epochs (used to shape
                the cosine decay after warmup).
            min_lr: Minimum learning rate the schedule will decay to.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self, epoch):
        """
        Updates the optimizer's learning rate for the given epoch.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            float: The learning rate applied to the first param group,
            for logging purposes.
        """
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * factor)
        
        return self.optimizer.param_groups[0]['lr']


# ============================================================
# TOP-K METRICS WITH PER-SS STATISTICS
# ============================================================
def compute_topk_metrics_with_ss(logits_aa, logits_cluster_all, y_aa, y_cl, y_ss,
                                  aa_to_n_clusters, k_values=[1, 5, 10, 20],
                                  ss_names=None):
    """
    Computes Top-K accuracy for AA and Joint (AA+CL) predictions, plus a
    breakdown of the same statistics for each secondary structure (SS)
    type.
    
    Args:
        logits_aa: [B, n_aa] amino acid logits.
        logits_cluster_all: [B, n_aa, max_clusters] cluster logits.
        y_aa: [B] true amino acid indices.
        y_cl: [B] true cluster indices.
        y_ss: [B] SS label for the central residue.
        aa_to_n_clusters: Dict {aa_idx: n_clusters}.
        k_values: List of k values to compute top-k accuracy for.
        ss_names: dict mapping {ss_idx: "name"} (e.g. {0: "Helix",
            1: "Sheet", ...}). Defaults to generic "SS_<i>" labels.
    
    Returns:
        dict with keys:
        - topk_aa: {1: acc1, 5: acc5, ...} (as top{k}_aa entries)
        - topk_joint: {1: acc1, 5: acc5, ...} (as top{k}_joint entries)
        - acc_cl_given_true_aa: cluster accuracy given the true AA
        - ss_stats: {ss_name: {topk_aa: {...}, topk_joint: {...}, count: N}}
    """
    device = logits_aa.device
    B = logits_aa.size(0)
    
    if B == 0:
        return {f'top{k}_aa': 0.0 for k in k_values} | \
               {f'top{k}_joint': 0.0 for k in k_values} | \
               {'acc_cl_given_true_aa': 0.0, 'ss_stats': {}}
    
    y_aa = y_aa.to(device).long()
    y_cl = y_cl.to(device).long()
    y_ss = y_ss.to(device).long()
    
    if ss_names is None:
        ss_names = {i: f"SS_{i}" for i in range(8)}
    
    results = {}
    
    # Group samples by SS type
    ss_groups = defaultdict(list)
    for i in range(B):
        ss_idx = int(y_ss[i].item())
        ss_groups[ss_idx].append(i)
    
    # Top-K AA accuracy (global)
    for k in k_values:
        if k > logits_aa.size(1):
            k = logits_aa.size(1)
        
        _, topk_preds = torch.topk(logits_aa, k, dim=1)
        correct = (topk_preds == y_aa.unsqueeze(1)).any(dim=1)
        results[f'top{k}_aa'] = correct.float().mean().item()
    
    # Cluster accuracy given the TRUE AA (global)
    correct_cl = 0
    valid_samples = 0
    
    for i in range(B):
        aa_idx = int(y_aa[i].item())
        if aa_idx not in aa_to_n_clusters:
            continue
        
        n_cl = aa_to_n_clusters[aa_idx]
        if n_cl <= 0:
            continue
        
        logits_cl = logits_cluster_all[i, aa_idx, :n_cl]
        pred_cl = torch.argmax(logits_cl).item()
        true_cl = int(y_cl[i].item())
        
        if true_cl < n_cl:
            valid_samples += 1
            if pred_cl == true_cl:
                correct_cl += 1
    
    results['acc_cl_given_true_aa'] = correct_cl / max(1, valid_samples)
    
    # Top-K Joint (AA+CL) accuracy (global)
    for k in k_values:
        correct_joint = 0
        
        for i in range(B):
            true_aa = int(y_aa[i].item())
            true_cl = int(y_cl[i].item())
            
            scores = []
            for aa_idx in range(logits_aa.size(1)):
                if aa_idx not in aa_to_n_clusters:
                    continue
                
                n_cl = aa_to_n_clusters[aa_idx]
                logit_aa_i = logits_aa[i, aa_idx].item()
                
                for cl_idx in range(n_cl):
                    logit_cl_i = logits_cluster_all[i, aa_idx, cl_idx].item()
                    score = logit_aa_i + logit_cl_i
                    scores.append((score, aa_idx, cl_idx))
            
            scores.sort(reverse=True, key=lambda x: x[0])
            
            topk_pairs = [(aa, cl) for _, aa, cl in scores[:k]]
            if (true_aa, true_cl) in topk_pairs:
                correct_joint += 1
        
        results[f'top{k}_joint'] = correct_joint / B
    
    # PER-SS STATISTICS
    ss_stats = {}
    
    for ss_idx, sample_indices in ss_groups.items():
        if len(sample_indices) == 0:
            continue
        
        ss_name = ss_names.get(ss_idx, f"SS_{ss_idx}")
        ss_stats[ss_name] = {'count': len(sample_indices)}
        
        # Top-K AA for this SS type
        for k in k_values:
            if k > logits_aa.size(1):
                k = logits_aa.size(1)
            
            correct_ss = 0
            for i in sample_indices:
                _, topk_preds = torch.topk(logits_aa[i], k)
                if y_aa[i] in topk_preds:
                    correct_ss += 1
            
            ss_stats[ss_name][f'top{k}_aa'] = correct_ss / len(sample_indices)
        
        # Top-K Joint for this SS type
        for k in k_values:
            correct_joint_ss = 0
            
            for i in sample_indices:
                true_aa = int(y_aa[i].item())
                true_cl = int(y_cl[i].item())
                
                scores = []
                for aa_idx in range(logits_aa.size(1)):
                    if aa_idx not in aa_to_n_clusters:
                        continue
                    
                    n_cl = aa_to_n_clusters[aa_idx]
                    logit_aa_i = logits_aa[i, aa_idx].item()
                    
                    for cl_idx in range(n_cl):
                        logit_cl_i = logits_cluster_all[i, aa_idx, cl_idx].item()
                        score = logit_aa_i + logit_cl_i
                        scores.append((score, aa_idx, cl_idx))
                
                scores.sort(reverse=True, key=lambda x: x[0])
                
                topk_pairs = [(aa, cl) for _, aa, cl in scores[:k]]
                if (true_aa, true_cl) in topk_pairs:
                    correct_joint_ss += 1
            
            ss_stats[ss_name][f'top{k}_joint'] = correct_joint_ss / len(sample_indices)
        
        # Cluster accuracy for this SS type
        correct_cl_ss = 0
        valid_samples_ss = 0
        
        for i in sample_indices:
            aa_idx = int(y_aa[i].item())
            if aa_idx not in aa_to_n_clusters:
                continue
            
            n_cl = aa_to_n_clusters[aa_idx]
            if n_cl <= 0:
                continue
            
            logits_cl = logits_cluster_all[i, aa_idx, :n_cl]
            pred_cl = torch.argmax(logits_cl).item()
            true_cl = int(y_cl[i].item())
            
            if true_cl < n_cl:
                valid_samples_ss += 1
                if pred_cl == true_cl:
                    correct_cl_ss += 1
        
        ss_stats[ss_name]['acc_cl_given_true_aa'] = correct_cl_ss / max(1, valid_samples_ss)
    
    results['ss_stats'] = ss_stats
    
    return results


# ============================================================
# LIGHTWEIGHT AA-ONLY METRICS (for the training loop)
# ============================================================
def compute_topk_aa_only(logits_aa, y_aa, k_values=[1, 5, 10]):
    """
    Lightweight, fully vectorized counterpart of
    compute_topk_metrics_with_ss, computing ONLY the top-k accuracy for
    the amino acid prediction (no joint accuracy, no ss_stats).

    Mathematically identical to the initial (already vectorized) part of
    compute_topk_metrics_with_ss -- the AA top-k loop -- simply isolated
    into its own function, without the per-sample Python loops for joint
    accuracy and ss_stats, which are never saved or used during training
    (see train_metrics in train_single_fold).

    No per-sample GPU->CPU synchronization: the only implicit .item()
    call is the final one from .mean(), to get the Python scalar to log.

    Args:
        logits_aa: [B, n_aa] amino acid logits.
        y_aa: [B] true amino acid indices.
        k_values: List of k values to compute top-k accuracy for.

    Returns:
        dict with keys 'top{k}_aa' for each k in k_values, plus
        placeholder 'top{k}_joint' and 'acc_cl_given_true_aa' entries
        set to 0.0 (kept only for dict-shape compatibility with
        compute_topk_metrics_with_ss; not meaningful values).
    """
    B = logits_aa.size(0)
    if B == 0:
        return {f'top{k}_aa': 0.0 for k in k_values} | \
               {f'top{k}_joint': 0.0 for k in k_values} | \
               {'acc_cl_given_true_aa': 0.0}

    y_aa = y_aa.to(logits_aa.device).long()
    results = {}

    for k in k_values:
        k_eff = min(k, logits_aa.size(1))
        _, topk_preds = torch.topk(logits_aa, k_eff, dim=1)
        correct = (topk_preds == y_aa.unsqueeze(1)).any(dim=1)
        results[f'top{k}_aa'] = correct.float().mean().item()

    # Placeholders for compatibility with the original dict shape: during
    # training these values are never read (only the losses are), but we
    # leave them at 0.0 so any code expecting these keys doesn't break.
    for k in k_values:
        results[f'top{k}_joint'] = 0.0
    results['acc_cl_given_true_aa'] = 0.0

    return results


def _vectorized_loss_cl(logits_cluster_all, y_aa, y_cl, aa_to_n_clusters, loss_fn_cl, device):
    """
    Vectorized (per-AA-group, not per-sample) computation of loss_cl,
    mathematically identical to the original loop:

        for i in range(B):
            aa_idx = int(y_aa[i].item())
            ...
            logits_i = logits_cluster_all[i, aa_idx, :n_cl].unsqueeze(0)
            target_i = clamp(y_cl[i], 0, n_cl-1)
            loss_cl_list.append(loss_fn_cl(logits_i, target_i))
        loss_cl = mean(loss_cl_list)

    Instead of iterating over every sample in the batch (up to 32
    iterations, each with its own .item() synchronization), samples are
    grouped by aa_idx (at most ~20 groups, typically fewer) and the loss
    is computed in a batched way per group. The final result (a
    count-weighted mean, equal to a plain mean over valid samples) has
    the same formula as the original, down to floating-point precision
    of torch's reduction.

    Args:
        logits_cluster_all: [B, n_aa, max_clusters] cluster logits.
        y_aa: [B] true amino acid indices.
        y_cl: [B] true cluster indices.
        aa_to_n_clusters: Dict {aa_idx: n_clusters}.
        loss_fn_cl: A CrossEntropyLoss (or similar) callable used per
            group.
        device: Torch device to place the zero-loss fallback tensor on.

    Returns:
        torch.Tensor: Scalar cluster loss, averaged over all valid
        samples in the batch (0.0 if no amino acid in the batch has any
        valid cluster).
    """
    unique_aas = torch.unique(y_aa)
    weighted_losses = []
    counts = []

    for aa_idx_t in unique_aas:
        aa_idx = aa_idx_t.item()  # at most ~20 syncs per batch, not B
        n_cl = aa_to_n_clusters.get(aa_idx, 0)
        if n_cl <= 0:
            continue

        mask = (y_aa == aa_idx_t)
        group_logits = logits_cluster_all[mask, aa_idx, :n_cl]      # [n_group, n_cl]
        group_targets = y_cl[mask].clamp(0, n_cl - 1)               # [n_group]

        group_loss = loss_fn_cl(group_logits, group_targets)        # mean over the group
        n_group = mask.sum()

        weighted_losses.append(group_loss * n_group.float())
        counts.append(n_group.float())

    if not weighted_losses:
        return torch.tensor(0.0, device=device)

    return torch.stack(weighted_losses).sum() / torch.stack(counts).sum()


# ============================================================
# TRAINING WITH SS STATS
# ============================================================
def train_epoch_v2(model, loader, opt, aa_to_n_clusters, 
                   lambda_cl=0.001, grad_clip=1.0, device='cuda',
                   label_smoothing=0.1, use_mixup=False, mixup_alpha=0.2,
                   ss_names=None):
    """
    Runs one training epoch.

    Args:
        model: The ProteinMLM_v2 model being trained.
        loader: DataLoader providing the training batches.
        opt: Optimizer used to update the model's parameters.
        aa_to_n_clusters: Dict {aa_idx: n_clusters}.
        lambda_cl: Weight applied to the cluster loss term.
        grad_clip: Max gradient norm for gradient clipping (0 to
            disable).
        device: Torch device to train on.
        label_smoothing: Label smoothing factor for the AA loss (the
            cluster loss uses half this value).
        use_mixup: Whether to randomly apply mixup augmentation to each
            batch.
        mixup_alpha: Alpha parameter of the mixup Beta distribution.
        ss_names: Unused here (kept for interface symmetry with
            eval_epoch_v2); no per-SS statistics are computed during
            training.

    Returns:
        dict with keys 'loss', 'loss_aa', 'loss_cl', 'loss_total', and
        'top{k}_aa'/'top{k}_joint' for k in [1, 5, 10] (the joint entries
        are placeholders, see `compute_topk_aa_only`), averaged over all
        batches in the epoch.
    """
    model.train()
    
    loss_fn_aa = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_fn_cl = nn.CrossEntropyLoss(label_smoothing=label_smoothing * 0.5)
    
    total_loss_aa  = 0.0
    total_loss_cl  = 0.0
    total_loss     = 0.0
    all_metrics = {f'top{k}_aa': 0.0 for k in [1, 5, 10]}
    all_metrics.update({f'top{k}_joint': 0.0 for k in [1, 5, 10]})
    n_batches = 0
    
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        B = batch['aa'].size(0)
        
        mixup_lambda = None
        mixup_indices = None
        if use_mixup and np.random.random() > 0.5:
            mixup_lambda = np.random.beta(mixup_alpha, mixup_alpha)
            mixup_indices = torch.randperm(B, device=device)
        
        logits_aa, logits_cluster_all, valid_mask = model(
            batch, mixup_lambda=mixup_lambda, mixup_indices=mixup_indices
        )
        
        y_aa = batch['y_aa'].to(device)
        y_cl = batch['y_cl'].to(device)
        
        if mixup_lambda is not None:
            loss_aa = mixup_lambda * loss_fn_aa(logits_aa, y_aa) + \
                      (1 - mixup_lambda) * loss_fn_aa(logits_aa, y_aa[mixup_indices])
        else:
            loss_aa = loss_fn_aa(logits_aa, y_aa)
        
        loss_cl = _vectorized_loss_cl(
            logits_cluster_all, y_aa, y_cl, aa_to_n_clusters, loss_fn_cl, device
        )
        loss = loss_aa + lambda_cl * loss_cl
        
        opt.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        
        total_loss_aa += loss_aa.item()  # only loss_aa is logged, consistent with validation
        total_loss_cl += loss_cl.item()
        total_loss    += loss.item()
        
        # Lightweight metrics (AA only, no joint/ss_stats): during training
        # these extra values are never saved or read (see train_metrics in
        # train_single_fold) -- computing them used to cost a per-sample
        # Python loop on every batch, avoided here.
        with torch.no_grad():
            metrics = compute_topk_aa_only(
                logits_aa, y_aa, k_values=[1, 5, 10]
            )
            for k in all_metrics:
                all_metrics[k] += metrics[k]
        
        n_batches += 1
    
    total_loss_aa /= n_batches
    total_loss_cl /= n_batches
    total_loss    /= n_batches
    for k in all_metrics:
        all_metrics[k] /= n_batches
    
    return {'loss': total_loss_aa, 'loss_aa': total_loss_aa, 'loss_cl': total_loss_cl, 'loss_total': total_loss, **all_metrics}


def eval_epoch_v2(model, loader, aa_to_n_clusters, lambda_cl=0.001, device='cuda',
                  ss_names=None, verbose_ss=False):
    """
    Runs one evaluation epoch with full per-SS statistics.

    Args:
        model: The ProteinMLM_v2 model being evaluated.
        loader: DataLoader providing the evaluation batches.
        aa_to_n_clusters: Dict {aa_idx: n_clusters}.
        lambda_cl: Weight applied to the cluster loss term.
        device: Torch device to evaluate on.
        ss_names: Optional dict mapping SS class index to a display
            name, used to label the per-SS breakdown.
        verbose_ss: If True, print a per-SS-type summary table after
            evaluation.

    Returns:
        dict with keys 'loss', 'loss_aa', 'loss_cl', 'loss_total',
        'top{k}_aa'/'top{k}_joint' for k in [1, 5, 10, 20],
        'acc_cl_given_true_aa', and 'ss_stats' (a per-SS-type breakdown
        of the same metrics), all averaged over the batches in the
        epoch.
    """
    model.eval()
    
    loss_fn_aa = nn.CrossEntropyLoss()
    loss_fn_cl = nn.CrossEntropyLoss()
    
    total_loss_aa  = 0.0
    total_loss_cl  = 0.0
    total_loss     = 0.0
    all_metrics = {f'top{k}_aa': 0.0 for k in [1, 5, 10, 20]}
    all_metrics.update({f'top{k}_joint': 0.0 for k in [1, 5, 10, 20]})
    all_metrics['acc_cl_given_true_aa'] = 0.0
    
    # Accumulate SS stats
    ss_stats_accum = defaultdict(lambda: defaultdict(list))
    
    n_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            logits_aa, logits_cluster_all, valid_mask = model(batch)
            
            y_aa = batch['y_aa'].to(device)
            y_cl = batch['y_cl'].to(device)
            
            # Extract the central residue's SS
            L = batch['aa'].size(1)
            center = L // 2
            y_ss = batch['ss'][:, center].to(device)
            
            loss_aa = loss_fn_aa(logits_aa, y_aa)
            
            B = logits_aa.size(0)
            loss_cl = _vectorized_loss_cl(
                logits_cluster_all, y_aa, y_cl, aa_to_n_clusters, loss_fn_cl, device
            )
            loss = loss_aa + lambda_cl * loss_cl
            
            total_loss_aa += loss_aa.item()
            total_loss_cl += loss_cl.item()
            total_loss    += loss.item()
            
            # Compute metrics WITH SS stats
            metrics = compute_topk_metrics_with_ss(
                logits_aa, logits_cluster_all, y_aa, y_cl, y_ss, aa_to_n_clusters,
                k_values=[1, 5, 10, 20], ss_names=ss_names
            )
            
            for k in all_metrics:
                if k != 'ss_stats':
                    all_metrics[k] += metrics[k]
            
            # Accumulate SS stats
            for ss_name, ss_dict in metrics['ss_stats'].items():
                for metric_name, value in ss_dict.items():
                    if metric_name != 'count':
                        ss_stats_accum[ss_name][metric_name].append(value)
                    else:
                        if 'count' not in ss_stats_accum[ss_name]:
                            ss_stats_accum[ss_name]['count'] = 0
                        ss_stats_accum[ss_name]['count'] += value
            
            n_batches += 1
    
    total_loss_aa /= n_batches
    total_loss_cl /= n_batches
    total_loss    /= n_batches
    for k in all_metrics:
        all_metrics[k] /= n_batches
    
    # Average the accumulated SS stats
    ss_stats_final = {}
    for ss_name, metrics_dict in ss_stats_accum.items():
        ss_stats_final[ss_name] = {'count': metrics_dict['count']}
        for metric_name, values in metrics_dict.items():
            if metric_name != 'count':
                ss_stats_final[ss_name][metric_name] = sum(values) / len(values)
    
    all_metrics['ss_stats'] = ss_stats_final
    
    # Print SS stats if requested
    if verbose_ss and ss_stats_final:
        print("\n  Secondary Structure Statistics:")
        for ss_name in sorted(ss_stats_final.keys()):
            ss_dict = ss_stats_final[ss_name]
            print(f"    {ss_name:12s} (n={ss_dict['count']:4.0f}): "
                  f"Top1={ss_dict.get('top1_aa', 0):.3f} "
                  f"Top5={ss_dict.get('top5_aa', 0):.3f} "
                  f"Top10={ss_dict.get('top10_aa', 0):.3f} | "
                  f"Joint Top5={ss_dict.get('top5_joint', 0):.3f}")
    
    return {'loss': total_loss_aa, 'loss_aa': total_loss_aa, 'loss_cl': total_loss_cl, 'loss_total': total_loss, **all_metrics}


# ============================================================
# MAIN TRAINING v2
# ============================================================
def train_mlm_v2(args):
    """
    Main training routine: loads the datasets, builds the model and
    optimizer, runs the training loop with early stopping on
    validation top-5 accuracy, then evaluates the best checkpoint on
    the held-out test set and saves the results (history, SS stats,
    best model checkpoint) under `args.outdir`.

    Args:
        args: Parsed argparse Namespace with all training hyperparameters
            (see the CLI argument definitions in `__main__` below).

    Returns:
        Path: Path to the saved best-model checkpoint
        (`<outdir>/best_model.pt`).
    """
    from dataset import Windows3Dataset, make_loaders_with_exposed_val
    from rotamer_library import load_rotamer_library
    from torch.optim import AdamW
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    rotlib = load_rotamer_library(args.rotamer_csv)
    
    ds = Windows3Dataset(args.data, rotamer_lib=rotlib, mask_central=True, seed=args.seed)
    ds_test = Windows3Dataset(args.test_data, rotamer_lib=rotlib, mask_central=True, seed=args.seed)
    
    # Get the SS names mapping
    ss_names = getattr(ds, 'ss_names', {i: f"SS_{i}" for i in range(8)})
    print(f"\nSecondary Structure Types: {ss_names}")
    
    train_loader, val_loader, test_loader = make_loaders_with_exposed_val(
        ds, ds_test, 
        val_size=args.val_size,
        batch_size=args.batch_size, 
        seed=args.seed,
        sasa_threshold=args.sasa_threshold
    )
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    print("\n" + "="*60)
    print("INITIALIZING PROTEIN MLM v2")
    print("="*60)
    
    aa_to_n_clusters = ds.get_aa_to_n_clusters()
    
    model = ProteinMLM_v2(
        n_aa=ds.n_aa - 1,
        aa_to_n_clusters=aa_to_n_clusters,
        n_pos=ds.n_pos,
        n_ss=ds.n_ss,
        d_embed=args.d_embed,
        d_hidden=args.d_hidden,
        n_lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        embed_dropout=args.embed_dropout
    ).to(device)
    
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = WarmupCosineScheduler(
        opt, 
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.lr * 0.01
    )
    
    print("\n" + "="*60)
    print("TRAINING START")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Mixup: {args.use_mixup} (alpha={args.mixup_alpha})")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print("="*60)
    
    best_top5 = 0.0
    patience_counter = 0
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': [], 'lr': [],
               'train_top1_aa': [], 'val_top1_aa': [],
               'train_top5_aa': [], 'val_top5_aa': [],
               'train_top10_aa': [], 'val_top10_aa': [],
               'val_ss_stats': []}  # History of per-epoch SS stats
    
    for epoch in range(1, args.epochs + 1):
        current_lr = scheduler.step(epoch - 1)
        
        train_metrics = train_epoch_v2(
            model, train_loader, opt, aa_to_n_clusters,
            lambda_cl=args.lambda_cl, grad_clip=args.grad_clip, device=device,
            label_smoothing=args.label_smoothing,
            use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
            ss_names=ss_names
        )
        
        # Show SS stats every 10 epochs during validation
        verbose_ss = (epoch % 10 == 0 or epoch == args.epochs)
        
        val_metrics = eval_epoch_v2(
            model, val_loader, aa_to_n_clusters,
            lambda_cl=args.lambda_cl, device=device,
            ss_names=ss_names, verbose_ss=verbose_ss
        )
        
        print(f"[{epoch:03d}] lr={current_lr:.2e} | Train: L={train_metrics['loss']:.3f} "
              f"Top1={train_metrics['top1_aa']:.3f} Top5={train_metrics['top5_aa']:.3f}")
        print(f"       Val: L={val_metrics['loss']:.3f} "
              f"Top1={val_metrics['top1_aa']:.3f} "
              f"Top5={val_metrics['top5_aa']:.3f} "
              f"Top10={val_metrics['top10_aa']:.3f}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['lr'].append(current_lr)
        history['train_top1_aa'].append(train_metrics['top1_aa'])
        history['val_top1_aa'].append(val_metrics['top1_aa'])
        history['train_top5_aa'].append(train_metrics['top5_aa'])
        history['val_top5_aa'].append(val_metrics['top5_aa'])
        history['train_top10_aa'].append(train_metrics['top10_aa'])
        history['val_top10_aa'].append(val_metrics['top10_aa'])
        history['val_ss_stats'].append(val_metrics.get('ss_stats', {}))
        
        # Early stopping based on top-5
        if val_metrics['top5_aa'] > best_top5:
            best_top5 = val_metrics['top5_aa']
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            torch.save({
                'model': model.state_dict(),
                'meta': {'n_aa': ds.n_aa, 'aa_to_n_clusters': aa_to_n_clusters, 
                        'ss_names': ss_names, 'args': vars(args)},
                'epoch': epoch
            }, outdir / 'best_model.pt')
            
            print(f"       Best (Top-5={best_top5:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    with open(outdir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Test
    print("\n" + "="*60)
    print("TEST EVALUATION")
    print("="*60)
    
    ckpt = torch.load(outdir / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    
    # Test with verbose SS stats
    test_metrics = eval_epoch_v2(
        model, test_loader, aa_to_n_clusters,
        lambda_cl=args.lambda_cl, device=device,
        ss_names=ss_names, verbose_ss=True
    )
    
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS:")
    print(f"  Top-1  AA:  {test_metrics['top1_aa']:.3f} ({test_metrics['top1_aa']*100:.1f}%)")
    print(f"  Top-5  AA:  {test_metrics['top5_aa']:.3f} ({test_metrics['top5_aa']*100:.1f}%)")
    print(f"  Top-10 AA:  {test_metrics['top10_aa']:.3f} ({test_metrics['top10_aa']*100:.1f}%)")
    print(f"  Top-20 AA:  {test_metrics['top20_aa']:.3f} ({test_metrics['top20_aa']*100:.1f}%)")
    print(f"\n  Top-5  Joint:  {test_metrics['top5_joint']:.3f} ({test_metrics['top5_joint']*100:.1f}%)")
    print(f"  Top-10 Joint:  {test_metrics['top10_joint']:.3f} ({test_metrics['top10_joint']*100:.1f}%)")
    print(f"  Cluster (given true AA): {test_metrics['acc_cl_given_true_aa']:.3f}")
    print("="*60)
    
    # Save detailed SS stats
    if 'ss_stats' in test_metrics:
        with open(outdir / 'test_ss_stats.json', 'w') as f:
            json.dump(test_metrics['ss_stats'], f, indent=2)
        
        print("\nTest SS statistics saved to test_ss_stats.json")
    
    return outdir / 'best_model.pt'


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Train the ProteinMLM_v2 BiLSTM model on 5-residue windows (AA + rotamer cluster + SASA + SS)")
    ap.add_argument("--data", required=True,
                    help="Path to the training/validation dataset (.npz file).")
    ap.add_argument("--test_data", required=True,
                    help="Path to a separate held-out test dataset (.npz file).")
    ap.add_argument("--rotamer_csv", default="rotamer_localmax_maxdens.csv",
                    help="Path to the rotamer library CSV used to build/validate cluster assignments (default: rotamer_localmax_maxdens.csv).")
    ap.add_argument("--outdir", default="runs/mlm_topk",
                    help="Directory where the best checkpoint, training history, and test SS stats will be written (default: runs/mlm_topk).")
    
    ap.add_argument("--epochs", type=int, default=150,
                    help="Maximum number of training epochs (default: 150).")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="Number of windows per training/evaluation batch (default: 32).")
    ap.add_argument("--val_size", type=int, default=146,
                    help="Number of exposed windows to hold out for validation (default: 146).")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Initial learning rate for the AdamW optimizer (default: 1e-3).")
    ap.add_argument("--weight_decay", type=float, default=1e-4,
                    help="Weight decay (L2 regularization) for the AdamW optimizer (default: 1e-4).")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="Maximum gradient norm for gradient clipping (default: 1.0).")
    ap.add_argument("--patience", type=int, default=30,
                    help="Number of epochs without validation top-5 improvement before early stopping (default: 30).")
    
    ap.add_argument("--d_embed", type=int, default=128,
                    help="Embedding dimension for each input feature (amino acid, cluster, SASA, SS, position) (default: 128).")
    ap.add_argument("--d_hidden", type=int, default=256,
                    help="Hidden size of each LSTM direction (default: 256).")
    ap.add_argument("--lstm_layers", type=int, default=2,
                    help="Number of stacked LSTM layers (default: 2).")
    ap.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout probability used in the LSTM, attention pooling, and classification heads (default: 0.2).")
    ap.add_argument("--lambda_cl", type=float, default=1.0,
                    help="Weight applied to the cluster loss term (default: 1.0).")
    
    ap.add_argument("--embed_dropout", type=float, default=0.3,
                    help="Dropout probability applied to each input embedding before concatenation (default: 0.3).")
    ap.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing factor for the amino acid loss; the cluster loss uses half this value (default: 0.1).")
    ap.add_argument("--use_mixup", action="store_true",
                    help="Enable mixup data augmentation during training.")
    ap.add_argument("--mixup_alpha", type=float, default=0.2,
                    help="Alpha parameter of the mixup Beta distribution, used only if --use_mixup is set (default: 0.2).")
    ap.add_argument("--warmup_epochs", type=int, default=5,
                    help="Number of learning-rate warmup epochs before the cosine schedule begins (default: 5).")
    
    ap.add_argument("--sasa_threshold", type=float, default=0.1199,
                    help="SASA threshold to define exposed residues")
    
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed used for reproducibility (data split, weight init, mixup, etc.) (default: 42).")
    ap.add_argument("--cpu", action="store_true",
                    help="Force training on CPU even if a CUDA GPU is available.")
    
    args = ap.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    best = train_mlm_v2(args)
    print(f"\nBest model: {best}")
