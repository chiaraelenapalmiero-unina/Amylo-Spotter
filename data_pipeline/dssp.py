#DSSP per struttura secondaria e accessibilità al solvente
#tipico utilizzo per DSSP


from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
p = PDBParser()
structure = p.get_structure("2rnm_filtered_new_r5c23d4.8a0.0_new", "./2rnm_filtered_new_r5c23d4.8a0.0_new.pdb")
model = structure[0]
dssp = DSSP(model, './2rnm_filtered_new_r5c23d4.8a0.0_new.pdb', dssp='/home/cpalmiero/.conda/envs/dssp/bin/mkdssp')
for key in dssp.keys():
    a_key = key[0], key[1]
    print(dssp[a_key])  #stampa tuple con informazioni su ogni residuo