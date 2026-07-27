#distribuzione residui interni o esterni, leggendo dal file dssp.csv
#!/usr/bin/env python3
"""
Script per estrarre i valori di ACC_norm dai CSV generati da batch_dssp_parse
per tutte le strutture secondarie, plotterne la densità (KDE) senza bin
e calcolare i minimi locali della densità.
Supporta colonne che contengono 'acc_norm', incluso 'ACC_norm %'.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


def main(csv_path, n_points=1000):
    df = pd.read_csv(csv_path)

    columns = [c for c in df.columns if 'acc_norm' in c.lower()]
    acc_col = columns[0]

    # Prendi tutti i valori senza filtrare per la colonna 'SS'
    values = df[acc_col].values
    
    print(f"Processati {len(values)} residui totali.")

    # Calcola ACC% normalizzata se necessario
    if 'percent' not in acc_col.lower():
        df['ACC_percent'] = df[acc_col]
    else:
        df['ACC_percent'] = df[acc_col]

    # Estrai i valori di ACC_percent
    values = df['ACC_percent'].values

    # Costruisci una KDE continua
    kde = gaussian_kde(values)
    xs = np.linspace(0, values.max(), n_points)
    ys = kde(xs)

    # Trova i minimi locali (inverte ys per trovare picchi)
    minima_idx, _ = find_peaks(-ys)
    minima = xs[minima_idx]
    print(f"Minimi locali della densità: {np.round(minima, 2)}")

    # Plot densità (KDE) senza bin
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, label='KDE', linewidth=3)
    # Evidenzia i minimi locali
    plt.scatter(minima, ys[minima_idx], color='red', marker='x', label='local minima', s=100)
    # Limita l'asse x ai valori validi
    plt.xlim(left=0)
    plt.xlabel('RSA [%]', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.title(f'RSA distribution (all SS)', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot KDE e calcolo minimi locali di ACC% normalizzata per tutte le SS"
    )
    parser.add_argument('csv_path', help='Percorso al file dssp_summary.csv')
    parser.add_argument('--points', '-p', type=int, default=1000,
                        help="Numero di punti per KDE (default: 1000)")
    args = parser.parse_args()
    main(args.csv_path, n_points=args.points)