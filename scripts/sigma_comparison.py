#!/usr/bin/env python
"""
Calculate implied cross sections for first 10 rows of Set3.
"""

# Pre-extracted first 10 rows from data2018_shuffled_3.csv
rows = [
    {'ZeeRaw': 63, 'ZeeEffAComb': 0.2098, 'ZeeLumi': 2.9211,
     'ZmumuRaw': 96, 'ZmumuEffAComb': 0.3028, 'ZmumuLumi': 2.7943},
    {'ZeeRaw': 234, 'ZeeEffAComb': 0.2108, 'ZeeLumi': 10.7146,
     'ZmumuRaw': 327, 'ZmumuEffAComb': 0.2879, 'ZmumuLumi': 9.9482},
    {'ZeeRaw': 372, 'ZeeEffAComb': 0.1957, 'ZeeLumi': 18.7331,
     'ZmumuRaw': 548, 'ZmumuEffAComb': 0.2866, 'ZmumuLumi': 16.3188},
    {'ZeeRaw': 194, 'ZeeEffAComb': 0.2135, 'ZeeLumi': 8.7594,
     'ZmumuRaw': 258, 'ZmumuEffAComb': 0.2998, 'ZmumuLumi': 7.6582},
    {'ZeeRaw': 359, 'ZeeEffAComb': 0.2110, 'ZeeLumi': 16.6597,
     'ZmumuRaw': 525, 'ZmumuEffAComb': 0.3022, 'ZmumuLumi': 17.1822},
]

# Need more rows - let me load from CSV
import pandas as pd

df = pd.read_csv(r"D:\HEP\ATLAS\LIV\Set3\data2018_shuffled_3.csv", nrows=10)

sigma_theory = 1.97  # nb
f_bkg = 0.005

print("Implied Cross Sections for First 10 Rows")
print("=" * 90)
print(f"{'Row':>3} | {'σ_ee (nb)':>10} | {'Diff_ee':>10} | {'σ_μμ (nb)':>10} | {'Diff_μμ':>10}")
print("-" * 90)

for i, row in df.iterrows():
    # σ_implied = N × (1 - f_bkg) / (L_inst × EffAComb × t)
    # where L_inst = ZeeLumi (instantaneous), t = LBLive
    sigma_ee = row['ZeeRaw'] * (1 - f_bkg) / (row['ZeeLumi'] * row['ZeeEffAComb'] * row['LBLive'])
    sigma_mumu = row['ZmumuRaw'] * (1 - f_bkg) / (row['ZmumuLumi'] * row['ZmumuEffAComb'] * row['LBLive'])
    
    diff_ee = (sigma_ee / sigma_theory - 1) * 100
    diff_mumu = (sigma_mumu / sigma_theory - 1) * 100
    
    print(f"{i:>3} | {sigma_ee:>10.4f} | {diff_ee:>+9.1f}% | {sigma_mumu:>10.4f} | {diff_mumu:>+9.1f}%")

print("-" * 90)

# Compute averages
sigma_ee_avg = (df['ZeeRaw'] * (1 - f_bkg) / (df['ZeeLumi'] * df['ZeeEffAComb'] * df['LBLive'])).mean()
sigma_mumu_avg = (df['ZmumuRaw'] * (1 - f_bkg) / (df['ZmumuLumi'] * df['ZmumuEffAComb'] * df['LBLive'])).mean()

print(f"Avg | {sigma_ee_avg:>10.4f} | {(sigma_ee_avg/sigma_theory-1)*100:>+9.1f}% | {sigma_mumu_avg:>10.4f} | {(sigma_mumu_avg/sigma_theory-1)*100:>+9.1f}%")
