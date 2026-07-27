# rotamer_library.py
"""
Utility per caricare la libreria di rotameri e creare mapping AA -> clusters validi
"""
import pandas as pd
import torch
from pathlib import Path

# Mapping standard 20 AA (ordine alfabetico)
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA20)}

# Mapping 3-letter -> 1-letter
AA3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

class RotamerLibrary:
    """
    Carica la libreria di rotameri e fornisce mapping AA -> cluster validi.
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self._build_mappings()
    
    def _build_mappings(self):
        """Costruisce tutti i mapping necessari"""
        # 1. AA_idx -> lista di cluster_idx validi
        self.aa_to_clusters = {}
        
        # 2. Numero di cluster per ogni AA
        self.aa_to_n_clusters = {}
        
        # 3. Mapping globale (aa_idx, cluster_local) -> cluster_global_idx
        #    dove cluster_local = 0,1,2,... per ogni AA
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
            
            # Ottieni tutti i cluster per questo AA (ordinati per cluster_id)
            aa_clusters = self.df[self.df['resname'] == aa_3letter].sort_values('cluster_id')
            n_clusters = len(aa_clusters)
            
            # Salva mapping
            self.aa_to_n_clusters[aa_idx] = n_clusters
            cluster_list = []
            
            for local_idx, (_, row) in enumerate(aa_clusters.iterrows()):
                cluster_list.append(local_idx)
                
                # Mapping bidirezionale
                self.pair_to_global[(aa_idx, local_idx)] = global_cluster_idx
                self.global_to_pair[global_cluster_idx] = (aa_idx, local_idx)
                
                global_cluster_idx += 1
            
            self.aa_to_clusters[aa_idx] = cluster_list
        
        self.n_valid_pairs = global_cluster_idx
        print(f"[ROTLIB] Loaded {len(self.aa_to_clusters)} AAs with {self.n_valid_pairs} total valid pairs")
        
        # Stampa sommario
        for aa_idx in sorted(self.aa_to_clusters.keys()):
            aa_name = AA20[aa_idx]
            n_cl = self.aa_to_n_clusters[aa_idx]
            print(f"  {aa_name}: {n_cl} cluster(s)")
    
    def get_valid_clusters(self, aa_idx: int):
        """Ritorna lista di cluster validi (locali) per un dato AA"""
        return self.aa_to_clusters.get(aa_idx, [])
    
    def get_n_clusters(self, aa_idx: int):
        """Ritorna numero di cluster per un dato AA"""
        return self.aa_to_n_clusters.get(aa_idx, 0)
    
    def get_max_clusters(self):
        """Ritorna il numero massimo di cluster per un singolo AA"""
        return max(self.aa_to_n_clusters.values()) if self.aa_to_n_clusters else 0
    
    def pair_to_global_idx(self, aa_idx: int, cluster_local: int):
        """Converte (aa, cluster_local) -> indice globale"""
        return self.pair_to_global.get((aa_idx, cluster_local), -1)
    
    def global_idx_to_pair(self, global_idx: int):
        """Converte indice globale -> (aa, cluster_local)"""
        return self.global_to_pair.get(global_idx, (-1, -1))
    
    def create_validity_mask(self, n_aa: int, max_clusters: int = None):
        """
        Crea maschera booleana [n_aa, max_clusters] dove True = coppia valida.
        
        Args:
            n_aa: numero totale di aminoacidi (di solito 20)
            max_clusters: se None, usa self.get_max_clusters()
        
        Returns:
            torch.BoolTensor di shape [n_aa, max_clusters]
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
        """Ottieni informazioni su un rotamero specifico"""
        row = self.df[(self.df['resname'] == aa_3letter) & 
                      (self.df['cluster_id'] == cluster_id)]
        if len(row) == 0:
            return None
        return row.iloc[0].to_dict()


def load_rotamer_library(csv_path: str = "rotamer_localmax_maxdens.csv"):
    """Helper per caricare la libreria"""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Rotamer library not found: {csv_path}")
    return RotamerLibrary(csv_path)


# Test
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
    
    # Esempi
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