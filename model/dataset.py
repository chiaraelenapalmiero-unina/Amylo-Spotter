# dataset.py - VERSIONE CON SASA CONTINUA E CON AGGIUNTA SS
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from pathlib import Path

class Windows3Dataset(Dataset):
    def __init__(self, 
                 npz_path: str, 
                 rotamer_lib=None,
                 mask_central: bool = True, 
                 rand_mask_prob: float = 0.0,
                 filter_center_exposure: int = None,
                 seed: int = 42):
        pack = np.load(npz_path)
        self.aa = pack["aa_ids"].astype(np.int64)
        self.cl = pack["cluster_idx"].astype(np.int64)
        
        # Carica SASA
        if "sasa" in pack:
            self.sasa = pack["sasa"].astype(np.float32)
            print(f"[DATASET] Loaded SASA values (range: {self.sasa.min():.3f}-{self.sasa.max():.3f})")
        elif "exposure" in pack:
            print("[DATASET] Warning: Using binary exposure as SASA approximation")
            self.sasa = pack["exposure"].astype(np.float32)
        else:
            raise ValueError("NPZ must contain either 'sasa' or 'exposure' field!")
        
        # 🆕 CARICA SECONDARY STRUCTURE
        if "ss" in pack:
            self.ss = pack["ss"].astype(np.int64)  # Assumo sia già codificata come indici
            self.n_ss = int(self.ss.max()) + 1  # Numero di classi SS
            print(f"[DATASET] Loaded Secondary Structure (classes: {self.n_ss})")
        else:
            # Se non presente, crea SS "unknown" (classe 0)
            print("[DATASET] Warning: No SS field found, using default class 0")
            self.ss = np.zeros_like(self.aa, dtype=np.int64)
            self.n_ss = 1
        
        self.pos = pack["pos_idx"].astype(np.int64)
        
        # ... resto del codice invariato ...
        
        self.rotamer_lib = rotamer_lib
        L = self.aa.shape[1]
        assert L % 2 == 1, f"Attesa finestra dispari, trovato L={L}"
        self.center = L // 2
        
        # Target del centrale
        self.y_aa = self.aa[:, self.center].copy()
        self.y_cl = self.cl[:, self.center].copy()
        self.y_sasa = self.sasa[:, self.center].copy()
        self.y_ss = self.ss[:, self.center].copy()  # 🆕 Target SS
        
        # Filtra campioni invalidi
        valid_targets = self.y_aa < 20
        
        if filter_center_exposure is not None:
            center_exposure = (self.y_sasa >= 0.2).astype(np.int64)
            exposure_mask = center_exposure == filter_center_exposure
            valid_targets = valid_targets & exposure_mask
            exposure_name = 'buried' if filter_center_exposure == 0 else 'exposed'
            print(f"[FILTER] Keeping only samples with center exposure={exposure_name}")
        
        if not valid_targets.all():
            n_invalid = (~valid_targets).sum()
            print(f"[FIX] Rimosso {n_invalid} campione(i) dopo filtri.")
            
            self.aa = self.aa[valid_targets]
            self.cl = self.cl[valid_targets]
            self.sasa = self.sasa[valid_targets]
            self.ss = self.ss[valid_targets]  # 🆕 Filtra SS
            self.pos = self.pos[valid_targets]
            self.y_aa = self.y_aa[valid_targets]
            self.y_cl = self.y_cl[valid_targets]
            self.y_sasa = self.y_sasa[valid_targets]
            self.y_ss = self.y_ss[valid_targets]  # 🆕 Filtra target SS
        
        # Vocabolari
        n_aa_vocab = int(self.aa.max()) + 1
        n_cl_vocab = int(self.cl.max()) + 1
        
        self.aa_mask_idx = n_aa_vocab
        self.cl_mask_idx = n_cl_vocab
        
        # Maschera centrale
        if mask_central:
            self.aa[:, self.center] = self.aa_mask_idx
            self.cl[:, self.center] = self.cl_mask_idx
            # NON maskare SASA e SS - mantieni valori reali come features
        
        # Random masking
        if rand_mask_prob > 0.0:
            rng = np.random.default_rng(seed)
            m = rng.random(self.aa.shape) < rand_mask_prob
            self.aa = np.where(m, self.aa_mask_idx, self.aa)
            self.cl = np.where(m, self.cl_mask_idx, self.cl)
            if mask_central:
                self.aa[:, self.center] = self.aa_mask_idx
                self.cl[:, self.center] = self.cl_mask_idx
        
        # Vocabolari finali
        self.n_aa = n_aa_vocab + 1
        self.n_cl = n_cl_vocab + 1
        self.n_pos = int(self.pos.max()) + 1
        
        if self.rotamer_lib is not None:
            self._validate_with_library()
    
    def _validate_with_library(self):
        """Valida che i target siano coppie valide secondo la libreria"""
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
            print(f"[OK] All {len(self)} samples have valid (AA, cluster) pairs ✓")
    
    def get_aa_to_n_clusters(self):
        """Ritorna dizionario {aa_idx: n_clusters} dalla rotamer library"""
        if self.rotamer_lib is None:
            raise ValueError("RotamerLibrary non fornita al dataset!")
        return self.rotamer_lib.aa_to_n_clusters.copy()
    
    def get_valid_mask(self):
        """Ritorna maschera di validità [n_aa, max_clusters]"""
        if self.rotamer_lib is None:
            raise ValueError("RotamerLibrary non fornita!")
        max_cl = self.rotamer_lib.get_max_clusters()
        return self.rotamer_lib.create_validity_mask(self.n_aa - 1, max_cl)
    
    def __len__(self):
        return self.aa.shape[0]
    
    def __getitem__(self, idx):
        return {
            "aa": torch.from_numpy(self.aa[idx]),
            "cl": torch.from_numpy(self.cl[idx]),
            "sasa": torch.from_numpy(self.sasa[idx]),
            "ss": torch.from_numpy(self.ss[idx]),  # 🆕 Aggiungi SS
            "pos": torch.from_numpy(self.pos[idx]),
            "y_aa": torch.tensor(int(self.y_aa[idx]), dtype=torch.long),
            "y_cl": torch.tensor(int(self.y_cl[idx]), dtype=torch.long),
            "y_sasa": torch.tensor(float(self.y_sasa[idx]), dtype=torch.float32),
            "y_ss": torch.tensor(int(self.y_ss[idx]), dtype=torch.long),  # 🆕 Target SS
        }



def make_loaders_with_exposed_val(ds_train, ds_test, val_size=146, batch_size=32, seed=42, sasa_threshold=0.1199):
    """
    Crea train/val/test loaders dove validation contiene solo finestre EXPOSED.
    
    🆕 Usa SASA per determinare exposure: SASA >= threshold = exposed
    
    Args:
        ds_train: Dataset completo
        ds_test: Test set separato
        val_size: Numero di finestre exposed per validation
        batch_size: Batch size
        seed: Random seed
        sasa_threshold: Soglia SASA per considerare un residuo exposed (default 0.1199)
    """
    exposed_indices = []
    buried_indices = []
    
    for i in range(len(ds_train)):
        sample = ds_train[i]
        center_idx = sample['aa'].shape[0] // 2
        center_sasa = sample['sasa'][center_idx].item()
        
        # 🆕 Determina exposure da SASA
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
        print(f"  ⚠️  Warning: Richiesti {val_size} exposed, disponibili solo {len(exposed_indices)}")
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
    """Collate function per DataLoader"""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if torch.is_tensor(batch[0][key]):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    
    return collated


# Backward compatibility: mantieni le vecchie funzioni
def make_splits(ds: Windows3Dataset, train=0.8, val=0.1, test=0.1, seed=42):
    assert abs(train + val + test - 1.0) < 1e-8
    N = len(ds)
    n_train = int(N * train)
    n_val = int(N * val)
    n_test = N - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n_test], generator=g)