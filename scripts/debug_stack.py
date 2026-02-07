
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from run_batch_analysis import fold_phase, SIDEREAL_DAY_H

# Load one file
f = "output/scrambles_pq/scramble_0000.parquet"
df = pd.read_parquet(f)
print(f"Loaded {len(df)} rows.")

# Check OffLumi
n_valid = (df['OffLumi'] > 0).sum()
print(f"Valid OffLumi > 0: {n_valid}")

df_valid = df[df['OffLumi'] > 0].copy()

# Phase calc
per_h = SIDEREAL_DAY_H
per_sec = per_h * 3600.0
BIN_SEC = 840.0
nbins = int(np.round(per_sec / BIN_SEC))
print(f"nbins: {nbins}")

tmid = 0.5 * (df_valid['LBStart'].to_numpy() + df_valid['LBEnd'].to_numpy())
phi = fold_phase(tmid, per_h)

print(f"Phi range: {phi.min():.4f} - {phi.max():.4f}")
print(f"Phi mean: {phi.mean():.4f}")

idx = np.floor(phi * nbins).astype(int)
idx = np.clip(idx, 0, nbins - 1)

print("Index stats:")
print(pd.Series(idx).value_counts().sort_index())

# Check integration
n_total = np.zeros(nbins)
d_total = np.zeros(nbins)
np.add.at(d_total, idx, df_valid['OffLumi'].values)

print(f"Non-zero denominator bins: {(d_total > 0).sum()}")
