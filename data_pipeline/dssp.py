"""
DSSP for secondary structure and solvent accessibility.
Minimal example of typical DSSP usage via Biopython.

Before running, edit the variables below:
    PDB_ID       : Identifier used to label the parsed structure
                   (does not need to match the filename).
    PDB_FILE     : Path to the input .pdb file to analyze.
    DSSP_BINARY  : Path to the mkdssp executable on your system
                   (e.g. the output of `which mkdssp`).
"""

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

# --- Parameters to edit for your own run ---
PDB_ID = "2rnm_filtered_new_r5c23d4.8a0.0_new"
PDB_FILE = "./2rnm_filtered_new_r5c23d4.8a0.0_new.pdb"
DSSP_BINARY = "/home/cpalmiero/.conda/envs/dssp/bin/mkdssp"
# --------------------------------------------

p = PDBParser()
structure = p.get_structure(PDB_ID, PDB_FILE)
model = structure[0]
dssp = DSSP(model, PDB_FILE, dssp=DSSP_BINARY)

# Each key is (chain_id, residue_id); dssp[key] returns a tuple with
# per-residue information (residue index, amino acid, secondary structure,
# relative solvent accessibility, and further DSSP fields).
for key in dssp.keys():
    a_key = key[0], key[1]
    print(dssp[a_key])  # print the tuple of per-residue DSSP information
