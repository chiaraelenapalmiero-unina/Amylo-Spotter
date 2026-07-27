#!/usr/bin/env python3
"""
Script to run DSSP on all PDB files in a directory,
parse secondary structure and solvent accessibility for a specified chain,
and compute normalized ACC based on average SASA values.
Utilizza os.system per chiamare mkdssp come da terminale e gestisce errori di esecuzione.
Ora accetta un parametro --chain per indicare il chain ID.
"""
import os
import argparse
import pandas as pd

# Dictionary of average SASA values (Å²) per amino acid (3-letter code)
SASA_DICT = {'ALA':129.0,'ARG':274.0,'ASN':195.0,'ASP':193.0,'CYS':167.0,
           'GLN':223.0,'GLU':225.0,'GLY':104.0,'HIS':224.0,'ILE':197.0,
           'LEU':201.0,'LYS':236.0,'MET':224.0,'PHE':240.0,'PRO':159.0,
           'SER':155.0,'THR':172.0,'TRP':285.0,'TYR':263.0,'VAL':174.0
}
# Mapping from 1-letter to 3-letter codes
ONE_TO_THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

def run_dssp(pdb_path, dssp_path):
    """
    Esegue mkdssp usando os.system, come da terminale Linux:
      mkdssp -i input.pdb -o output.dssp
    Ritorna True se l'output è stato generato, False altrimenti.
    """
    cmd = f"mkdssp -i '{pdb_path}' -o '{dssp_path}'"
    rc = os.system(cmd)
    return rc == 0 and os.path.exists(dssp_path)


def parse_dssp_file(dssp_path, target_chain):
    """
    Parse a DSSP output file, estraendo per ogni residuo:
      - chain ID,
      - numero residuo,
      - amminoacido 3-lettere,
      - struttura secondaria,
      - ACC grezza e normalizzata.
    Filtra solo i residui della chain target_chain.
    """
    records = []
    start = False
    with open(dssp_path) as fh:
        for line in fh:
            if line.startswith("  #  RESIDUE"):
                start = True
                continue
            if not start:
                continue

            # Estrai chain ID (colonna 12, indice 11)
            chain_id = line[11].strip()
            if chain_id != target_chain:
                continue

            # Estrai campi utili
            res_num = line[5:10].strip()
            aa1 = line[13].strip()
            ss = line[16].strip()
            acc_str = line[35:38].strip()

            if aa1 == '!' or not acc_str:
                continue

            try:
                acc = int(acc_str)
            except ValueError:
                continue

            aa3 = ONE_TO_THREE.get(aa1)
            avg_sasa = SASA_DICT.get(aa3)
            if not aa3 or not avg_sasa:
                continue

            acc_norm = (acc / avg_sasa)*100

            acc_norm_0_1=(acc / avg_sasa)

            records.append({
                'PDB': os.path.basename(dssp_path).replace('.dssp',''),
                'Chain': chain_id,
                'ResNum': res_num,
                'AA': aa3,
                'SS': ss,
                'ACC_raw': acc,
                'ACC_norm%': round(acc_norm, 3),
                'ACC_norm' : round(acc_norm_0_1, 3)  })
    return records


def main(input_dir, output_csv, target_chain):
    """
    Processa tutti i .pdb in input_dir, genera .dssp,
    estrae dati per la chain target_chain e salva su CSV.
    Gestisce errori di creazione file DSSP.
    """
    all_data = []
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.pdb'):
            continue
        pdb_path = os.path.join(input_dir, fname)
        dssp_path = pdb_path.rsplit('.', 1)[0] + '.dssp'
        print(f"Processing {fname} (chain {target_chain})...")
        if not run_dssp(pdb_path, dssp_path):
            print(f"Warning: DSSP failed for {fname}, skipping.")
            continue
        try:
            records = parse_dssp_file(dssp_path, target_chain)
            all_data.extend(records)
        except FileNotFoundError:
            print(f"Warning: output file {dssp_path} non trovato, skip.")

    if not all_data:
        print(f"Nessun dato raccolto per chain {target_chain}")
        return
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch DSSP parsing e ACC normalization per chain specifica'
    )
    parser.add_argument(
        'input_dir', help='Directory with PDB files'
    )
    parser.add_argument(
        'output_csv', help='Path to output CSV file'
    )
    parser.add_argument(
        '--chain', '-c', default='E',
        help='Chain ID to filter (default: E)'
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_csv, args.chain)
