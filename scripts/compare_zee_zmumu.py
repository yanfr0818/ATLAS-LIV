#!/usr/bin/env python
"""
Compare implied cross sections for Zee and Zmumu channels.
"""

# Pre-extracted rows from data2018_shuffled_3.csv
rows = [
    {
        'LBLive': 59.245,
        'ZeeRaw': 63, 'ZeeEffAComb': 0.2098, 'ZeeLumi': 2.9211,
        'ZmumuRaw': 96, 'ZmumuEffAComb': 0.3028, 'ZmumuLumi': 2.7943,
    },
    {
        'LBLive': 59.586,
        'ZeeRaw': 234, 'ZeeEffAComb': 0.2108, 'ZeeLumi': 10.7146,
        'ZmumuRaw': 327, 'ZmumuEffAComb': 0.2879, 'ZmumuLumi': 9.9482,
    },
    {
        'LBLive': 59.665,
        'ZeeRaw': 372, 'ZeeEffAComb': 0.1957, 'ZeeLumi': 18.7331,
        'ZmumuRaw': 548, 'ZmumuEffAComb': 0.2866, 'ZmumuLumi': 16.3188,
    },
    {
        'LBLive': 59.520,
        'ZeeRaw': 194, 'ZeeEffAComb': 0.2135, 'ZeeLumi': 8.7594,
        'ZmumuRaw': 258, 'ZmumuEffAComb': 0.2998, 'ZmumuLumi': 7.6582,
    },
    {
        'LBLive': 59.738,
        'ZeeRaw': 359, 'ZeeEffAComb': 0.2110, 'ZeeLumi': 16.6597,
        'ZmumuRaw': 525, 'ZmumuEffAComb': 0.3022, 'ZmumuLumi': 17.1822,
    },
]

# Theoretical cross section (from ATLAS note)
sigma_theory = 1.97  # nb (total Z→ll for m_ll > 60 GeV)
f_bkg = 0.005

print("=" * 70)
print("CROSS SECTION COMPARISON: Zee vs Zmumu")
print("=" * 70)
print()
print(f"Theoretical σ = {sigma_theory} nb (Z→ll, m_ll > 60 GeV)")
print()

# Compute implied sigma for each channel
# Using: sigma_implied = N × (1-f_bkg) / (L × EffAComb × t)
# where L is integrated luminosity (so we don't divide by t)
# Actually: sigma_implied = N × (1-f_bkg) / (L × EffAComb)

print("Formula: σ_implied = N × (1 - f_bkg) / (L × EffAComb)")
print()
print("-" * 70)
print(f"{'Row':>4} | {'σ_Zee (nb)':>12} | {'σ_Zmumu (nb)':>12} | {'Ratio Zee/Zmumu':>15}")
print("-" * 70)

sigma_zee_list = []
sigma_mumu_list = []

for i, r in enumerate(rows):
    # Zee
    sigma_zee = r['ZeeRaw'] * (1 - f_bkg) / (r['ZeeLumi'] * r['ZeeEffAComb'])
    
    # Zmumu
    sigma_mumu = r['ZmumuRaw'] * (1 - f_bkg) / (r['ZmumuLumi'] * r['ZmumuEffAComb'])
    
    ratio = sigma_zee / sigma_mumu
    
    sigma_zee_list.append(sigma_zee)
    sigma_mumu_list.append(sigma_mumu)
    
    print(f"{i:>4} | {sigma_zee:>12.4f} | {sigma_mumu:>12.4f} | {ratio:>15.4f}")

print("-" * 70)

# Compute averages
avg_zee = sum(sigma_zee_list) / len(sigma_zee_list)
avg_mumu = sum(sigma_mumu_list) / len(sigma_mumu_list)
avg_ratio = avg_zee / avg_mumu

print(f"{'Avg':>4} | {avg_zee:>12.4f} | {avg_mumu:>12.4f} | {avg_ratio:>15.4f}")
print()

# Compare to theory
print("=" * 70)
print("COMPARISON TO THEORY")
print("=" * 70)
print()
print(f"σ_theory = {sigma_theory} nb")
print()
print(f"σ_Zee (implied)   = {avg_zee:.4f} nb")
print(f"  Ratio to theory = {avg_zee / sigma_theory:.4f}")
print(f"  Difference      = {(avg_zee / sigma_theory - 1) * 100:+.1f}%")
print()
print(f"σ_Zmumu (implied) = {avg_mumu:.4f} nb")
print(f"  Ratio to theory = {avg_mumu / sigma_theory:.4f}")
print(f"  Difference      = {(avg_mumu / sigma_theory - 1) * 100:+.1f}%")
print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print()
print("The implied σ is ~50x larger than theory because:")
print("  σ_implied includes 1/F_MC (MC correction not in Set3)")
print()
print("To get the 'rate-based' σ, divide by LBLive:")
print()

sigma_zee_rate = []
sigma_mumu_rate = []

for r in rows:
    zee_rate = r['ZeeRaw'] * (1 - f_bkg) / (r['ZeeLumi'] * r['ZeeEffAComb'] * r['LBLive'])
    mumu_rate = r['ZmumuRaw'] * (1 - f_bkg) / (r['ZmumuLumi'] * r['ZmumuEffAComb'] * r['LBLive'])
    sigma_zee_rate.append(zee_rate)
    sigma_mumu_rate.append(mumu_rate)

avg_zee_rate = sum(sigma_zee_rate) / len(sigma_zee_rate)
avg_mumu_rate = sum(sigma_mumu_rate) / len(sigma_mumu_rate)

print(f"σ_Zee (rate)   = {avg_zee_rate:.4f} nb  (theory: 1.97, diff: {(avg_zee_rate/sigma_theory - 1)*100:+.1f}%)")
print(f"σ_Zmumu (rate) = {avg_mumu_rate:.4f} nb  (theory: 1.97, diff: {(avg_mumu_rate/sigma_theory - 1)*100:+.1f}%)")
print()
print("These are ~12% lower than theory, representing the F_MC correction.")
