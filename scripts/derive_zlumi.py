#!/usr/bin/env python
"""
derive_zlumi.py - Derive and validate Z luminosity formulas from raw Set 3 data.

DISCOVERED FORMULAS:
    ZeeLumi = ZeeRaw / (sigma_Zee * ZeeEffAComb)
    ZmumuLumi = ZmumuRaw / (sigma_Zmumu * ZmumuEffAComb)
    ZllLumi = weighted average of ZeeLumi and ZmumuLumi (inverse variance)

Cross-section constants (derived from data):
    sigma_Zee ~ 102.7 pb
    sigma_Zmumu ~ 96.5 pb
"""

import pandas as pd
import numpy as np

DATA_PATH = r"Set3\data2018_shuffled_3.csv"

def derive_cross_sections(df):
    """Derive the effective cross-sections from the data."""
    # sigma = ZeeRaw / (ZeeLumi * ZeeEffAComb)
    sigma_zee = df['ZeeRaw'] / (df['ZeeLumi'] * df['ZeeEffAComb'])
    sigma_mumu = df['ZmumuRaw'] / (df['ZmumuLumi'] * df['ZmumuEffAComb'])
    
    return sigma_zee.mean(), sigma_zee.std(), sigma_mumu.mean(), sigma_mumu.std()


def validate_formula(df, sigma_zee, sigma_mumu):
    """Validate the derived formulas against actual values."""
    # Derive luminosities
    df['ZeeLumi_derived'] = df['ZeeRaw'] / (sigma_zee * df['ZeeEffAComb'])
    df['ZmumuLumi_derived'] = df['ZmumuRaw'] / (sigma_mumu * df['ZmumuEffAComb'])
    
    # Compute residuals
    zee_resid = (df['ZeeLumi_derived'] - df['ZeeLumi']) / df['ZeeLumi']
    mumu_resid = (df['ZmumuLumi_derived'] - df['ZmumuLumi']) / df['ZmumuLumi']
    
    return zee_resid, mumu_resid


def derive_zll_formula(df, sigma_zee, sigma_mumu):
    """Test how ZllLumi is combined from ZeeLumi and ZmumuLumi."""
    # Derive individual luminosities
    zee_lumi = df['ZeeRaw'] / (sigma_zee * df['ZeeEffAComb'])
    mumu_lumi = df['ZmumuRaw'] / (sigma_mumu * df['ZmumuEffAComb'])
    
    # Method 1: Simple average
    zll_avg = 0.5 * (zee_lumi + mumu_lumi)
    
    # Method 2: Inverse variance weighted average
    # Weight by 1/err^2
    w_zee = 1 / (df['ZeeLumiErr']**2)
    w_mumu = 1 / (df['ZmumuLumiErr']**2)
    zll_weighted = (w_zee * df['ZeeLumi'] + w_mumu * df['ZmumuLumi']) / (w_zee + w_mumu)
    
    # Method 3: Weighted using actual lumi values
    zll_weighted_derived = (w_zee * zee_lumi + w_mumu * mumu_lumi) / (w_zee + w_mumu)
    
    return zll_avg, zll_weighted, zll_weighted_derived


def main():
    print("=" * 60)
    print("Z LUMINOSITY FORMULA DERIVATION AND VALIDATION")
    print("=" * 60)
    print()
    
    print("Loading data from Set 3 (2018)...")
    df = pd.read_csv(DATA_PATH, nrows=500)
    print(f"Loaded {len(df)} rows")
    
    # Filter to valid rows
    df = df[(df['ZeeRaw'] > 0) & (df['ZmumuRaw'] > 0) & 
            (df['ZeeEffAComb'] > 0) & (df['ZmumuEffAComb'] > 0)].copy()
    print(f"Valid rows (all channels > 0): {len(df)}")
    print()
    
    # Step 1: Derive cross-sections
    print("=" * 60)
    print("STEP 1: DERIVING EFFECTIVE CROSS-SECTIONS")
    print("=" * 60)
    sigma_zee, sigma_zee_std, sigma_mumu, sigma_mumu_std = derive_cross_sections(df)
    
    print(f"sigma_Zee   = {sigma_zee:.4f} ± {sigma_zee_std:.4f} pb")
    print(f"sigma_Zmumu = {sigma_mumu:.4f} ± {sigma_mumu_std:.4f} pb")
    print()
    
    # Step 2: Validate formulas
    print("=" * 60)
    print("STEP 2: VALIDATING FORMULAS")
    print("=" * 60)
    print("Formula: L = N_raw / (sigma * EffAComb)")
    print()
    
    zee_resid, mumu_resid = validate_formula(df, sigma_zee, sigma_mumu)
    
    print("ZeeLumi reconstruction:")
    print(f"  Mean residual:  {zee_resid.mean()*100:+.4f}%")
    print(f"  Std residual:   {zee_resid.std()*100:.4f}%")
    print(f"  Max |residual|: {zee_resid.abs().max()*100:.4f}%")
    print()
    
    print("ZmumuLumi reconstruction:")
    print(f"  Mean residual:  {mumu_resid.mean()*100:+.4f}%")
    print(f"  Std residual:   {mumu_resid.std()*100:.4f}%")
    print(f"  Max |residual|: {mumu_resid.abs().max()*100:.4f}%")
    print()
    
    # Step 3: Derive ZllLumi formula
    print("=" * 60)
    print("STEP 3: DERIVING ZllLumi COMBINATION FORMULA")
    print("=" * 60)
    
    zll_avg, zll_weighted, zll_weighted_derived = derive_zll_formula(df, sigma_zee, sigma_mumu)
    
    # Test each method
    resid_avg = (zll_avg - df['ZllLumi']) / df['ZllLumi']
    resid_wtd = (zll_weighted - df['ZllLumi']) / df['ZllLumi']
    resid_wtd_der = (zll_weighted_derived - df['ZllLumi']) / df['ZllLumi']
    
    print("Method 1: Simple average of ZeeLumi + ZmumuLumi")
    print(f"  Mean residual: {resid_avg.mean()*100:+.4f}%, Std: {resid_avg.std()*100:.4f}%")
    print()
    
    print("Method 2: Inverse-variance weighted average (using actual Lumi)")
    print(f"  Mean residual: {resid_wtd.mean()*100:+.4f}%, Std: {resid_wtd.std()*100:.4f}%")
    print()
    
    print("Method 3: Inverse-variance weighted average (using derived Lumi)")
    print(f"  Mean residual: {resid_wtd_der.mean()*100:+.4f}%, Std: {resid_wtd_der.std()*100:.4f}%")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY: DERIVED FORMULAS")
    print("=" * 60)
    print()
    print("ZeeLumi   = ZeeRaw / (sigma_Zee * ZeeEffAComb)")
    print(f"          where sigma_Zee = {sigma_zee:.4f} pb")
    print()
    print("ZmumuLumi = ZmumuRaw / (sigma_Zmumu * ZmumuEffAComb)")
    print(f"          where sigma_Zmumu = {sigma_mumu:.4f} pb")
    print()
    print("ZllLumi   = (w_e * ZeeLumi + w_m * ZmumuLumi) / (w_e + w_m)")
    print("          where w_e = 1/ZeeLumiErr^2, w_m = 1/ZmumuLumiErr^2")
    print()
    print("Acceptance × Efficiency breakdown:")
    print(f"  ZeeEffAComb / ZeeEffComb   = {(df['ZeeEffAComb']/df['ZeeEffComb']).mean():.4f} (Acceptance)")
    print(f"  ZmumuEffAComb / ZmumuEffComb = {(df['ZmumuEffAComb']/df['ZmumuEffComb']).mean():.4f} (Acceptance)")
    print()
    
    # Sample validation
    print("=" * 60)
    print("SAMPLE VALIDATION (first 5 rows)")
    print("=" * 60)
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        zee_der = row['ZeeRaw'] / (sigma_zee * row['ZeeEffAComb'])
        mumu_der = row['ZmumuRaw'] / (sigma_mumu * row['ZmumuEffAComb'])
        
        w_e = 1 / (row['ZeeLumiErr']**2)
        w_m = 1 / (row['ZmumuLumiErr']**2)
        zll_der = (w_e * zee_der + w_m * mumu_der) / (w_e + w_m)
        
        print(f"Row {i}: ZeeRaw={row['ZeeRaw']:.0f}, ZmumuRaw={row['ZmumuRaw']:.0f}")
        print(f"  ZeeLumi:   Derived={zee_der:.4f}, Actual={row['ZeeLumi']:.4f}, "
              f"Diff={(zee_der-row['ZeeLumi'])/row['ZeeLumi']*100:+.2f}%")
        print(f"  ZmumuLumi: Derived={mumu_der:.4f}, Actual={row['ZmumuLumi']:.4f}, "
              f"Diff={(mumu_der-row['ZmumuLumi'])/row['ZmumuLumi']*100:+.2f}%")
        print(f"  ZllLumi:   Derived={zll_der:.4f}, Actual={row['ZllLumi']:.4f}, "
              f"Diff={(zll_der-row['ZllLumi'])/row['ZllLumi']*100:+.2f}%")
        print()


if __name__ == "__main__":
    main()
