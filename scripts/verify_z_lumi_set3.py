#!/usr/bin/env python
"""
verify_z_lumi_set3.py

Verifies the Z-based luminosity columns (ZeeLumi, ZmumuLumi) in Set 3 data
by recalculating them from raw counts and efficiency factors.

Method:
    L_Z = N * (1 - f_bkg) / (sigma_theory * eff_total * t)

Where:
    sigma_theory = 1.97 nb
    f_bkg = 0.005
    eff_total = EffAComb * F_MC(mu)
    t = LBLive
"""

import pandas as pd
import numpy as np
import os

# Total cross section theory
SIGMA_THEORY = 1.97  # nb
F_BKG = 0.005 # 0.5% background

def fmc_zee(mu, year):
    """F_MC for Z->ee channel (from sigma_with_fmc.py)"""
    if year in [2015, 2016]:
        return 0.907628 - 0.000328652*mu - 3.0512e-06*mu*mu
    elif year == 2017:
        return 0.904096 - 0.000172139*mu - 4.35328e-06*mu*mu
    elif year == 2018:
        return 0.90238 - 8.75767e-05*mu - 5.79201e-06*mu*mu
    else:
        raise ValueError(f"Unknown year: {year}")

def fmc_zmumu(mu, year):
    """F_MC for Z->mumu channel (from sigma_with_fmc.py)"""
    if year in [2015, 2016]:
        return 9.90074e-01 - 5.34716e-06*mu - 3.23366e-06*mu*mu
    elif year == 2017:
        return 9.91619e-01 - 1.21674e-04*mu - 1.58362e-06*mu*mu
    elif year == 2018:
        return 9.90808e-01 - 9.99749e-05*mu - 1.40241e-06*mu*mu
    else:
        raise ValueError(f"Unknown year: {year}")

def verify_lumi(df, year):
    print(f"Verifying Z Luminosity for Year {year}")
    print("=" * 140)
    print(f"{'Row':>3} | {'Type':>5} | {'Raw':>6} | {'Eff':>7} | {'F_MC':>7} | {'L_given':>10} | {'L_calc':>10} | {'Diff(%)':>9} | {'L_err_given':>12} | {'L_err_calc':>12}")
    print("-" * 140)

    for i, row in df.iterrows():
        mu = row['OffMu']
        lblive = row['LBLive']

        # ---------------------------------------------------------
        # Z -> ee
        # ---------------------------------------------------------
        zee_raw = row['ZeeRaw']
        zee_eff = row['ZeeEffAComb']
        zee_lumi_given = row['ZeeLumi']
        zee_lumi_err_given = row['ZeeLumiErr']
        
        fmc_e = fmc_zee(mu, year)
        
        # Calculate Lumi
        # Formula: L = N * (1-f_bkg) / (sigma * eff * fmc * t)
        if zee_raw > 0:
            zee_lumi_calc = zee_raw * (1 - F_BKG) / (SIGMA_THEORY * zee_eff * fmc_e * lblive)
            
            # Calculate Error
            # dL/L = sqrt( (dN/N)^2 + (dEff/Eff)^2 )
            # dN = sqrt(N) -> dN/N = 1/sqrt(N)
            rel_stat_err = 1.0 / np.sqrt(zee_raw)
            # ZeeErrAComb is absolute error.
            rel_eff_err = row['ZeeErrAComb'] / zee_eff
            total_rel_err = np.sqrt(rel_stat_err**2 + rel_eff_err**2) # ignoring sys err on sigma/fmc
            
            zee_lumi_err_calc = zee_lumi_calc * total_rel_err
            
            diff_e = (zee_lumi_calc - zee_lumi_given) / zee_lumi_given * 100
        else:
            zee_lumi_calc = 0
            zee_lumi_err_calc = 0
            diff_e = 0

        print(f"{i:>3} | {'Zee':>5} | {zee_raw:>6.0f} | {zee_eff:>7.4f} | {fmc_e:>7.4f} | {zee_lumi_given:>10.4f} | {zee_lumi_calc:>10.4f} | {diff_e:>+9.2f} | {zee_lumi_err_given:>12.4f} | {zee_lumi_err_calc:>12.4f}")

        # ---------------------------------------------------------
        # Z -> mumu
        # ---------------------------------------------------------
        zmumu_raw = row['ZmumuRaw']
        zmumu_eff = row['ZmumuEffAComb']
        zmumu_lumi_given = row['ZmumuLumi']
        zmumu_lumi_err_given = row['ZmumuLumiErr']

        fmc_m = fmc_zmumu(mu, year)

        if zmumu_raw > 0:
            zmumu_lumi_calc = zmumu_raw * (1 - F_BKG) / (SIGMA_THEORY * zmumu_eff * fmc_m * lblive)
            
            rel_stat_err = 1.0 / np.sqrt(zmumu_raw)
            rel_eff_err = row['ZmumuErrAComb'] / zmumu_eff
            total_rel_err = np.sqrt(rel_stat_err**2 + rel_eff_err**2)
            
            zmumu_lumi_err_calc = zmumu_lumi_calc * total_rel_err
            
            diff_m = (zmumu_lumi_calc - zmumu_lumi_given) / zmumu_lumi_given * 100
        else:
            zmumu_lumi_calc = 0
            zmumu_lumi_err_calc = 0
            diff_m = 0

        print(f"{' ':>3} | {'Zmumu':>5} | {zmumu_raw:>6.0f} | {zmumu_eff:>7.4f} | {fmc_m:>7.4f} | {zmumu_lumi_given:>10.4f} | {zmumu_lumi_calc:>10.4f} | {diff_m:>+9.2f} | {zmumu_lumi_err_given:>12.4f} | {zmumu_lumi_err_calc:>12.4f}")
        print("-" * 140)

if __name__ == "__main__":
    # Test on 2018 data
    # Path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "Set3", "data2018_shuffled_3.csv")
    
    print(f"Loading first 10 rows from {data_path}...")
    try:
        df = pd.read_csv(data_path, nrows=10)
        verify_lumi(df, 2018)
    except FileNotFoundError:
        print(f"Error: Could not find file {data_path}")
        print("Please check the path or run from correct directory.")
