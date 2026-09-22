# dataset.py - VERSION WITH CONTINUOUS SASA AND ADDED SS
"""
PyTorch Dataset and DataLoader utilities for 5-residue windows with
continuous SASA (solvent accessibility) and secondary structure (SS)
features, used to train the amino-acid/rotamer-cluster masked-language
model.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from pathlib import Path

class Windows3Dataset(Dataset):
    """
    Dataset of 5-residue windows loaded from a .npz file, with the
    central residue's amino acid and rotamer cluster masked out as the
    prediction target (masked-language-model style).

    Expected .npz fields:
        aa_ids     : [N, L] amino acid indices per residue.
        cluster_idx: [N, L] rotamer cluster indices per residue.
        pos_idx    : [N, L] position-in-window indices per residue.
        sasa       : [N, L] continuous SASA values (preferred), or
        exposure   : [N, L] binary exposure values, used as a fallback
                     approximation of SASA if "sasa" is absent.
        ss         : [N, L] secondary structure class indices (optional;
                     if absent, all residues are assigned class 0).
    """
    def __init__(self, 
                 npz_path: str, 
                 rotamer_lib=None,
                 mask_central: bool = True, 
                 rand_mask_prob: float = 0.0,
                 filter_center_exposure: int = None,
                 seed: int = 42):
        """
        Args:
            npz_path: Path to the .npz dataset file (see the class
                docstring for the expected fields).
            rotamer_lib: Optional RotamerLibrary instance. If provided,
                it is used to validate that every (AA, cluster) target
                pair is a known valid pair, and to build the AA ->
                n_clusters mapping via `get_aa_to_n_clusters`.
            mask_central: If True (default), mask out the central
                residue's amino acid and cluster index (replacing them
                with dedicated MASK indices) so the model has to predict
                them; SASA and SS at the center are left un-masked and
                usable as input features.
            rand_mask_prob: Additional per-residue random masking
                probability applied on top of `mask_central` (0.0 =
                disabled).
            filter_center_exposure: If set to 0 or 1, keep only samples
                whose central residue's exposure (SASA >= 0.2 threshold)
                equals this value (0 = buried, 1 = exposed). If None
                (default), no exposure-based filtering is applied.
            seed: Random seed used for the random masking.
        """
        pack = np.load(npz_path)
        self.aa = pack["aa_ids"].astype(np.int64)
        self.cl = pack["cluster_idx"].astype(np.int64)
        
        # Load SASA
        if "sasa" in pack:
            self.sasa = pack["sasa"].astype(np.float32)
            print(f"[DATASET] Loaded SASA values (range: {self.sasa.min():.3f}-{self.sasa.max():.3f})")
        elif "exposure" in pack:
            print("[DATASET] Warning: Using binary exposure as SASA approximation")
            self.sasa = pack["exposure"].astype(np.float32)
        else:
            raise ValueError("NPZ must contain either 'sasa' or 'exposure' field!")
        
        # Load Secondary Structure
        if "ss" in pack:
            self.ss = pack["ss"].astype(np.int64)  # Assumed already encoded as class indices
            self.n_ss = int(self.ss.max()) + 1  # Number of SS classes
            print(f"[DATASET] Loaded Secondary Structure (classes: {self.n_ss})")
        else:
            # If absent, create an "unknown" SS class (class 0)
            print("[DATASET] Warning: No SS field found, using default class 0")
            self.ss = np.zeros_like(self.aa, dtype=np.int64)
            self.n_ss = 1
        
        self.pos = pack["pos_idx"].astype(np.int64)
        
        # ... rest of the code unchanged ...
        
        self.rotamer_lib = rotamer_lib
        L = self.aa.shape[1]
        assert L % 2 == 1, f"Expected an odd-sized window, got L={L}"
        self.center = L // 2
        
        # Central-residue targets
        self.y_aa = self.aa[:, self.center].copy()
        self.y_cl = self.cl[:, self.center].copy()
        self.y_sasa = self.sasa[:, self.center].copy()
        self.y_ss = self.ss[:, self.center].copy()  # SS target
        
        # Filter out invalid samples
        valid_targets = self.y_aa < 20
        
        if filter_center_exposure is not None:
            center_exposure = (self.y_sasa >= 0.2).astype(np.int64)
            exposure_mask = center_exposure == filter_center_exposure
            valid_targets = valid_targets & exposure_mask
            exposure_name = 'buried' if filter_center_exposure == 0 else 'exposed'
            print(f"[FILTER] Keeping only samples with center exposure={exposure_name}")
        
        if not valid_targets.all():
            n_invalid = (~valid_targets).sum()
            print(f"[FIX] Removed {n_invalid} sample(s) after filtering.")
            
            self.aa = self.aa[valid_targets]
            self.cl = self.cl[valid_targets]
            self.sasa = self.sasa[valid_targets]
            self.ss = self.ss[valid_targets]  # Filter SS
            self.pos = self.pos[valid_targets]
            self.y_aa = self.y_aa[valid_targets]
            self.y_cl = self.y_cl[valid_targets]
            self.y_sasa = self.y_sasa[valid_targets]
            self.y_ss = self.y_ss[valid_targets]  # Filter SS target
        
        # Vocabularies
        n_aa_vocab = int(self.aa.max()) + 1
        n_cl_vocab = int(self.cl.max()) + 1
        
        self.aa_mask_idx = n_aa_vocab
        self.cl_mask_idx = n_cl_vocab
        
        # Mask the central residue
        if mask_central:
            self.aa[:, self.center] = self.aa_mask_idx
            self.cl[:, self.center] = self.cl_mask_idx
            # Do NOT mask SASA and SS - keep the real values as input features
        
        # Random masking
        if rand_mask_prob > 0.0:
            rng = np.random.default_rng(seed)
            m = rng.random(self.aa.shape) < rand_mask_prob
            self.aa = np.where(m, self.aa_mask_idx, self.aa)
            self.cl = np.where(m, self.cl_mask_idx, self.cl)
            if mask_central:
                self.aa[:, self.center] = self.aa_mask_idx
                self.cl[:, self.center] = self.cl_mask_idx
        
        # Final vocabulary sizes
        self.n_aa = n_aa_vocab + 1
        self.n_cl = n_cl_vocab + 1
        self.n_pos = int(self.pos.max()) + 1
        
        if self.rotamer_lib is not None:
            self._validate_with_library()
    
    def _validate_with_library(self):
        """
        Validates that every (AA, cluster) target pair in the dataset is
        a known valid pair according to `self.rotamer_lib`. Prints the
        first 5 invalid pairs found, as a warning, for quick debugging.
        """
        invalid_count = 0
        for i in range(len(self)):
            aa_idx = int(self.y_aa[i])
            cl_idx = int(self.y_cl[i])
            
            valid_clusters = self.rotamer_lib.get_valid_clusters(aa_idx)
            if cl_idx not in valid_clusters:
                invalid_count += 1
                if invalid_count <= 5:
                    print(f"[WARN] Sample {i}: AA={aa_idx} has cluster={cl_idx}, "
                          f"but valid clusters are {valid_clusters}")
        
        if invalid_count > 0:
            print(f"[WARN] Found {invalid_count}/{len(self)} samples with invalid (AA, cluster) pairs!")
        else:
            print(f"[OK] All {len(self)} samples have valid (AA, cluster) pairs")
    
    def get_aa_to_n_clusters(self):
        """
        Returns:
            dict: {aa_idx: n_clusters}, taken from the rotamer library.

        Raises:
            ValueError: If no `rotamer_lib` was provided to the dataset.
        """
        if self.rotamer_lib is None:
            raise ValueError("RotamerLibrary was not provided to the dataset!")
        return self.rotamer_lib.aa_to_n_clusters.copy()
    
    def get_valid_mask(self):
        """
        Returns:
            torch.BoolTensor: Validity mask of shape [n_aa, max_clusters],
            as built by the rotamer library (see
            `RotamerLibrary.create_validity_mask`).

        Raises:
            ValueError: If no `rotamer_lib` was provided to the dataset.
        """
        if self.rotamer_lib is None:
            raise ValueError("RotamerLibrary was not provided!")
        max_cl = self.rotamer_lib.get_max_clusters()
        return self.rotamer_lib.create_validity_mask(self.n_aa - 1, max_cl)
    
    def __len__(self):
        return self.aa.shape[0]
    
    def __getitem__(self, idx):
        return {
            "aa": torch.from_numpy(self.aa[idx]),
            "cl": torch.from_numpy(self.cl[idx]),
            "sasa": torch.from_numpy(self.sasa[idx]),
            "ss": torch.from_numpy(self.ss[idx]),  # SS feature
            "pos": torch.from_numpy(self.pos[idx]),
            "y_aa": torch.tensor(int(self.y_aa[idx]), dtype=torch.long),
            "y_cl": torch.tensor(int(self.y_cl[idx]), dtype=torch.long),
            "y_sasa": torch.tensor(float(self.y_sasa[idx]), dtype=torch.float32),
            "y_ss": torch.tensor(int(self.y_ss[idx]), dtype=torch.long),  # SS target
        }



def make_loaders_with_exposed_val(ds_train, ds_test, val_size=146, batch_size=32, seed=42, sasa_threshold=0.1199):
    """
    Builds train/val/test DataLoaders where the validation set contains
    only EXPOSED windows.

    Uses SASA to determine exposure: SASA >= threshold is considered
    exposed.

    Args:
        ds_train: The full training dataset (a Windows3Dataset), from
            which both the train and validation splits are drawn.
        ds_test: A separate, already-built test dataset.
        val_size: Number of exposed windows to allocate to the
            validation set (default: 146). If fewer exposed windows are
            available, all of them are used instead.
        batch_size: Batch size for all three DataLoaders.
        seed: Random seed used to shuffle the exposed/train indices.
        sasa_threshold: SASA threshold above which a residue is
            considered exposed (default: 0.1199).

    Returns:
        tuple (train_loader, val_loader, test_loader): Three
        torch.utils.data.DataLoader instances.
    """
    exposed_indices = []
    buried_indices = []
    
    for i in range(len(ds_train)):
        sample = ds_train[i]
        center_idx = sample['aa'].shape[0] // 2
        center_sasa = sample['sasa'][center_idx].item()
        
        # Determine exposure from SASA
        is_exposed = center_sasa >= sasa_threshold
        
        if is_exposed:
            exposed_indices.append(i)
        else:
            buried_indices.append(i)
    
    exposed_indices = np.array(exposed_indices)
    buried_indices = np.array(buried_indices)
    
    print(f"[SPLIT INFO] (SASA threshold={sasa_threshold})")
    print(f"  Total samples: {len(ds_train)}")
    print(f"  Exposed centers (SASA>={sasa_threshold}): {len(exposed_indices)}")
    print(f"  Buried centers (SASA<{sasa_threshold}): {len(buried_indices)}")
    
    # Shuffle exposed indices
    rng = np.random.RandomState(seed)
    rng.shuffle(exposed_indices)
    
    # Validation = exposed
    if len(exposed_indices) < val_size:
        print(f"  [WARN] Requested {val_size} exposed samples, only {len(exposed_indices)} available")
        val_size = len(exposed_indices)
    
    val_indices = exposed_indices[:val_size]
    remaining_exposed = exposed_indices[val_size:]
    
    # Train = remaining exposed + all buried
    train_indices = np.concatenate([remaining_exposed, buried_indices])
    rng.shuffle(train_indices)
    
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples (exposed only): {len(val_indices)}")
    print(f"  Test samples: {len(ds_test)}")
    
    # Create loaders
    train_subset = Subset(ds_train, train_indices.tolist())
    val_subset = Subset(ds_train, val_indices.tolist())
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """
    Collate function for the DataLoader: stacks tensor fields into a
    batch tensor, and leaves non-tensor fields as plain lists.

    Args:
        batch: List of per-sample dicts, as returned by
            `Windows3Dataset.__getitem__`.

    Returns:
        dict: Same keys as each sample, with tensor values stacked along
        a new batch dimension.
    """
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if torch.is_tensor(batch[0][key]):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    
    return collated


# Backward compatibility: keep the old functions
def make_splits(ds: Windows3Dataset, train=0.8, val=0.1, test=0.1, seed=42):
    """
    Splits a dataset into train/val/test subsets by fixed fractions.

    Args:
        ds: The Windows3Dataset (or any torch Dataset) to split.
        train: Fraction of samples to use for training (default: 0.8).
        val: Fraction of samples to use for validation (default: 0.1).
        test: Fraction of samples to use for testing (default: 0.1).
            train + val + test must sum to 1.0.
        seed: Random seed for the split (default: 42).

    Returns:
        tuple: Three torch.utils.data.Subset objects (train, val, test),
        as returned by `torch.utils.data.random_split`.
    """
    assert abs(train + val + test - 1.0) < 1e-8
    N = len(ds)
    n_train = int(N * train)
    n_val = int(N * val)
    n_test = N - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n_test], generator=g)
    return random_split(ds, [n_train, n_val, n_test], generator=g)
