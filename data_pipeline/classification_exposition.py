import pandas as pd
import sys

# usa l'argomento se presente, altrimenti il nome di default
csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dssp_classification.csv'
df = pd.read_csv(csv_path)

# se la colonna si chiama "ACC_norm %" invece di "ACC_norm", gestiscilo
metric = 'ACC_norm %' if 'ACC_norm %' in df.columns else 'ACC_norm'

# Soglia (modifica a piacere)
soglia = 0.12

# Classificazione per TUTTI i residui
df['Exposure'] = df[metric].apply(lambda x: 'buried' if x < soglia else 'exposed')

# Salva
df.to_csv('residui_classificati.csv', index=False)
print(f"File salvato come 'residui_classificati.csv'")
print(f"Totale residui processati: {len(df)}")
print(f"Residui buried: {(df['Exposure'] == 'buried').sum()}")
print(f"Residui exposed: {(df['Exposure'] == 'exposed').sum()}")