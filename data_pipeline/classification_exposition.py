"""
Classify residues as "buried" or "exposed" based on their normalized
solvent accessibility (ACC_norm), using a fixed threshold.

Usage:
    python classify_exposure.py [input_csv]

    input_csv : Optional path to the input CSV file, which must contain a
                normalized accessibility column named either "ACC_norm" or
                "ACC_norm %". Defaults to "dssp_classification.csv" if not
                provided.

Output:
    Writes "residui_classificati.csv" in the current directory, identical
    to the input file with an added "Exposure" column ("buried" or
    "exposed" for each residue).
"""
import pandas as pd
import sys

# Use the command-line argument if provided, otherwise fall back to the default filename
csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dssp_classification.csv'
df = pd.read_csv(csv_path)

# Handle both possible column names: "ACC_norm %" or "ACC_norm"
metric = 'ACC_norm %' if 'ACC_norm %' in df.columns else 'ACC_norm'

# Exposure threshold (edit as needed).
# Residues with a normalized accessibility below this value are
# classified as "buried"; at or above it, as "exposed".
threshold = 0.12

# Classify ALL residues
df['Exposure'] = df[metric].apply(lambda x: 'buried' if x < threshold else 'exposed')

# Save results
df.to_csv('residui_classificati.csv', index=False)
print(f"File saved as 'residui_classificati.csv'")
print(f"Total residues processed: {len(df)}")
print(f"Buried residues: {(df['Exposure'] == 'buried').sum()}")
print(f"Exposed residues: {(df['Exposure'] == 'exposed').sum()}")
