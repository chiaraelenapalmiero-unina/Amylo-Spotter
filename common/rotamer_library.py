# rotamer_library.py
"""
Utility for loading the rotamer library and building valid AA -> cluster mappings.

This module reads a CSV file describing, for each of the 20 standard amino acids,
the set of rotamer clusters observed for that residue type, and builds a set of
lookup structures (dicts and a boolean mask) that make it easy to:
  - go from an amino acid to its list of valid rotamer clusters
  - convert between a "local" cluster index (per amino acid) and a single
    "global" index that spans all (amino acid, cluster) pairs
  - build a validity mask usable directly in PyTorch models (e.g. to mask
    out invalid rotamer-cluster logits during training/inference)
"""
import pandas as pd
import torch
from pathlib import Path

# Mapping of the standard 20 amino acids (alphabetical order)
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA20)}

# Mapping from 3-letter to 1-letter amino acid codes
AA3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

class RotamerLibrary:
    """
    Loads the rotamer library and provides AA -> valid-cluster mappings.

    Expected CSV columns:
        resname    : 3-letter amino acid code (e.g. "ALA", "PHE")
        cluster_id : integer id of the rotamer cluster for that residue

    Any additional columns in the CSV (e.g. chi angles, density, etc.) are
    preserved in `self.df` and can be retrieved via `get_cluster_info`.
    """
    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: Path to the rotamer library CSV file. Must contain at
                least the `resname` and `cluster_id` columns described above.
        """
        self.df = pd.read_csv(csv_path)
        self._build_mappings()
    
    def _build_mappings(self):
        """Builds all the lookup mappings needed by the library."""
        # 1. aa_idx -> list of valid cluster_idx (local, per-AA indices)
        self.aa_to_clusters = {}
        
        # 2. Number of clusters for each AA
        self.aa_to_n_clusters = {}
        
        # 3. Global mapping (aa_idx, cluster_local) -> cluster_global_idx,
        #    where cluster_local = 0, 1, 2, ... for each AA
        self.pair_to_global = {}
        
        # 4. Reverse mapping: cluster_global_idx -> (aa_idx, cluster_local)
        self.global_to_pair = {}
        
        global_cluster_idx = 0
        
        for aa_3letter in sorted(self.df['resname'].unique()):
            if aa_3letter not in AA3_TO_1:
                print(f"[WARN] Skipping unknown AA: {aa_3letter}")
                continue
            
            aa_1letter = AA3_TO_1[aa_3letter]
            aa_idx = AA_TO_IDX[aa_1letter]
            
            # Get all clusters for this AA (sorted by cluster_id)
            aa_clusters = self.df[self.df['resname'] == aa_3letter].sort_values('cluster_id')
            n_clusters = len(aa_clusters)
            
            # Store mapping
            self.aa_to_n_clusters[aa_idx] = n_clusters
            cluster_list = []
            
            for local_idx, (_, row) in enumerate(aa_clusters.iterrows()):
                cluster_list.append(local_idx)
                
                # Bidirectional mapping
                self.pair_to_global[(aa_idx, local_idx)] = global_cluster_idx
                self.global_to_pair[global_cluster_idx] = (aa_idx, local_idx)
                
                global_cluster_idx += 1
            
            self.aa_to_clusters[aa_idx] = cluster_list
        
        self.n_valid_pairs = global_cluster_idx
        print(f"[ROTLIB] Loaded {len(self.aa_to_clusters)} AAs with {self.n_valid_pairs} total valid pairs")
        
        # Print summary
        for aa_idx in sorted(self.aa_to_clusters.keys()):
            aa_name = AA20[aa_idx]
            n_cl = self.aa_to_n_clusters[aa_idx]
            print(f"  {aa_name}: {n_cl} cluster(s)")
    
    def get_valid_clusters(self, aa_idx: int):
        """
        Args:
            aa_idx: Integer index of the amino acid (0-19, per AA20 order).

        Returns:
            List of valid local cluster indices for the given amino acid.
        """
        return self.aa_to_clusters.get(aa_idx, [])
    
    def get_n_clusters(self, aa_idx: int):
        """
        Args:
            aa_idx: Integer index of the amino acid (0-19, per AA20 order).

        Returns:
            Number of rotamer clusters available for the given amino acid.
        """
        return self.aa_to_n_clusters.get(aa_idx, 0)
    
    def get_max_clusters(self):
        """
        Returns:
            The maximum number of clusters found for any single amino acid
            (useful for sizing fixed-width tensors/masks).
        """
        return max(self.aa_to_n_clusters.values()) if self.aa_to_n_clusters else 0
    
    def pair_to_global_idx(self, aa_idx: int, cluster_local: int):
        """
        Args:
            aa_idx: Integer index of the amino acid (0-19, per AA20 order).
            cluster_local: Local cluster index for that amino acid
                (0, 1, 2, ... as assigned when the library was built).

        Returns:
            The global index of the (aa_idx, cluster_local) pair, or -1 if
            the pair is not valid/known.
        """
        return self.pair_to_global.get((aa_idx, cluster_local), -1)
    
    def global_idx_to_pair(self, global_idx: int):
        """
        Args:
            global_idx: Global cluster index, as returned by
                `pair_to_global_idx`.

        Returns:
            Tuple (aa_idx, cluster_local) corresponding to the given global
            index, or (-1, -1) if the index is not valid/known.
        """
        return self.global_to_pair.get(global_idx, (-1, -1))
    
    def create_validity_mask(self, n_aa: int, max_clusters: int = None):
        """
        Builds a boolean mask of shape [n_aa, max_clusters] where True marks
        a valid (amino acid, cluster) pair. Intended for masking invalid
        rotamer-cluster positions (e.g. in a model's output logits).

        Args:
            n_aa: Total number of amino acid types (usually 20).
            max_clusters: Number of columns in the mask. If None, defaults
                to `self.get_max_clusters()`.

        Returns:
            torch.BoolTensor of shape [n_aa, max_clusters].
        """
        if max_clusters is None:
            max_clusters = self.get_max_clusters()
        
        mask = torch.zeros(n_aa, max_clusters, dtype=torch.bool)
        
        for aa_idx in range(n_aa):
            valid_clusters = self.get_valid_clusters(aa_idx)
            if valid_clusters:
                mask[aa_idx, valid_clusters] = True
        
        return mask
    
    def get_cluster_info(self, aa_3letter: str, cluster_id: int):
        """
        Looks up the full row of data for one specific rotamer cluster.

        Args:
            aa_3letter: 3-letter amino acid code (e.g. "PHE").
            cluster_id: Cluster id as it appears in the source CSV
                (note: this is the original `cluster_id` column, not the
                local or global index used elsewhere in this class).

        Returns:
            Dict with all CSV columns for the matching row, or None if no
            matching row is found.
        """
        row = self.df[(self.df['resname'] == aa_3letter) & 
                      (self.df['cluster_id'] == cluster_id)]
        if len(row) == 0:
            return None
        return row.iloc[0].to_dict()


def load_rotamer_library(csv_path: str = "rotamer_localmax_maxdens.csv"):
    """
    Convenience helper to load the rotamer library.

    Args:
        csv_path: Path to the rotamer library CSV file. Defaults to
            "rotamer_localmax_maxdens.csv" in the current working directory.

    Returns:
        A RotamerLibrary instance built from the given CSV file.

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Rotamer library not found: {csv_path}")
    return RotamerLibrary(csv_path)


# Test / demo usage
if __name__ == "__main__":
    lib = load_rotamer_library()
    
    print("\n" + "="*60)
    print("VALIDITY MASK")
    print("="*60)
    
    mask = lib.create_validity_mask(20)
    print(f"Shape: {mask.shape}")
    print(f"Valid pairs: {mask.sum().item()}")
    
    print("\n" + "="*60)
    print("EXAMPLES")
    print("="*60)
    
    # Examples
    ala_idx = AA_TO_IDX['A']
    print(f"\nALA (idx={ala_idx}): {lib.get_n_clusters(ala_idx)} cluster(s)")
    print(f"  Valid clusters: {lib.get_valid_clusters(ala_idx)}")
    
    phe_idx = AA_TO_IDX['F']
    print(f"\nPHE (idx={phe_idx}): {lib.get_n_clusters(phe_idx)} cluster(s)")
    print(f"  Valid clusters: {lib.get_valid_clusters(phe_idx)}")
    
    # Global indexing
    print(f"\n" + "="*60)
    print("GLOBAL INDEXING EXAMPLES")
    print("="*60)
    print(f"(ALA=0, cluster=0) -> global_idx = {lib.pair_to_global_idx(0, 0)}")
    print(f"(PHE=5, cluster=3) -> global_idx = {lib.pair_to_global_idx(5, 3)}")
    print(f"global_idx=10 -> {lib.global_idx_to_pair(10)}")
