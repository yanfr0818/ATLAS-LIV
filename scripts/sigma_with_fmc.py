#!/usr/bin/env python
"""
Calculate implied cross sections with F_MC correction.
F_MC formulas from Uta, parameterized as function of pileup (mu).
"""
import pandas as pd

# F_MC formulas from Uta (quadratic in mu)
def fmc_zee(mu, year):
    """F_MC for Z->ee channel"""
    if year in [2015, 2016]:
        return 0.907628 - 0.000328652*mu - 3.0512e-06*mu*mu
    elif year == 2017:
        return 0.904096 - 0.000172139*mu - 4.35328e-06*mu*mu
    elif year == 2018:
        return 0.90238 - 8.75767e-05*mu - 5.79201e-06*mu*mu
    else:
        raise ValueError(f"Unknown year: {year}")

def fmc_zmumu(mu, year):
    """F_MC for Z->mumu channel"""
    if year in [2015, 2016]:
        return 9.90074e-01 - 5.34716e-06*mu - 3.23366e-06*mu*mu
    elif year == 2017:
        return 9.91619e-01 - 1.21674e-04*mu - 1.58362e-06*mu*mu
    elif year == 2018:
        return 9.90808e-01 - 9.99749e-05*mu - 1.40241e-06*mu*mu
    else:
        raise ValueError(f"Unknown year: {year}")


# Load data
df = pd.read_csv(r"D:\HEP\ATLAS\LIV\Set3\data2018_shuffled_3.csv", nrows=10)

sigma_theory = 1.97  # nb
f_bkg = 0.005
year = 2018

print("Implied Cross Sections WITH F_MC Correction (First 10 Rows)")
print("=" * 100)
print(f"Year: {year}, σ_theory = {sigma_theory} nb")
print()
print(f"{'Row':>3} | {'μ':>6} | {'F_ee':>6} | {'F_μμ':>6} | {'σ_ee':>8} | {'Diff_ee':>8} | {'σ_μμ':>8} | {'Diff_μμ':>8}")
print("-" * 100)

sigma_ee_list = []
sigma_mumu_list = []

for i, row in df.iterrows():
    mu = row['OffMu']
    fmc_ee = fmc_zee(mu, year)
    fmc_mm = fmc_zmumu(mu, year)
    
    # σ_implied = N × (1 - f_bkg) / (L_inst × EffAComb × F_MC × t)
    sigma_ee = row['ZeeRaw'] * (1 - f_bkg) / (row['ZeeLumi'] * row['ZeeEffAComb'] * fmc_ee * row['LBLive'])
    sigma_mumu = row['ZmumuRaw'] * (1 - f_bkg) / (row['ZmumuLumi'] * row['ZmumuEffAComb'] * fmc_mm * row['LBLive'])
    
    sigma_ee_list.append(sigma_ee)
    sigma_mumu_list.append(sigma_mumu)
    
    diff_ee = (sigma_ee / sigma_theory - 1) * 100
    diff_mumu = (sigma_mumu / sigma_theory - 1) * 100
    
    print(f"{i:>3} | {mu:>6.1f} | {fmc_ee:>6.4f} | {fmc_mm:>6.4f} | {sigma_ee:>8.4f} | {diff_ee:>+7.2f}% | {sigma_mumu:>8.4f} | {diff_mumu:>+7.2f}%")

print("-" * 100)

# Compute averages
avg_ee = sum(sigma_ee_list) / len(sigma_ee_list)
avg_mumu = sum(sigma_mumu_list) / len(sigma_mumu_list)

print(f"Avg |        |        |        | {avg_ee:>8.4f} | {(avg_ee/sigma_theory-1)*100:>+7.2f}% | {avg_mumu:>8.4f} | {(avg_mumu/sigma_theory-1)*100:>+7.2f}%")
print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Without F_MC: σ_ee ≈ 1.72 nb (-12.6%), σ_μμ ≈ 1.90 nb (-3.4%)")
print(f"With F_MC:    σ_ee ≈ {avg_ee:.4f} nb ({(avg_ee/sigma_theory-1)*100:+.1f}%), σ_μμ ≈ {avg_mumu:.4f} nb ({(avg_mumu/sigma_theory-1)*100:+.1f}%)")
