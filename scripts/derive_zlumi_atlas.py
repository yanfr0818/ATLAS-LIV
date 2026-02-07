#!/usr/bin/env python
"""
derive_zlumi_atlas.py

Cross-check Set 3 data using the official ATLAS Z-counting formula from
ATL-DAPR-PUB-2021-001 (ATLAS Note).

EQUATION (2) - Master Formula:
    L_Z = N_Z × (1 - f_bkg) / (σ_theory × A_MC × ε_T&P × F_MC × t)

Where:
    - N_Z = raw Z count per LB (ZeeRaw or ZmumuRaw in Set3)
    - f_bkg = 0.005 (background fraction from diboson + ttbar)
    - σ_theory = 1970 pb (inclusive Z→ll cross-section for m_ll > 60 GeV)
    - A_MC = acceptance factor (fiducial phase space)
    - ε_T&P = event-level efficiency from tag-and-probe
    - F_MC = pileup-dependent MC correction factor
    - t = LB duration (LBLive in Set3)

EQUATION (5) - Combined Efficiency:
    ε_Z = (1 - (1 - ε_trig)²) × ε_reco²

Set3 Column Mapping:
    - ZeeRaw → N_Z for electrons
    - ZeeEffTrig → ε_trig (trigger efficiency)
    - ZeeEffReco → ε_reco (reconstruction efficiency)
    - ZeeEffComb → ε_T&P (combined event-level efficiency)
    - ZeeEffAComb → A × ε_T&P (acceptance × efficiency)
    - ZeeLumi → L_Z (luminosity in nb^-1)
"""

import pandas as pd
import numpy as np

DATA_PATH = r"D:\HEP\ATLAS\LIV\Set3\data2018_shuffled_3.csv"

# Constants from ATLAS note
SIGMA_THEORY = 1970  # pb, inclusive Z→ll cross-section for m_ll > 60 GeV
F_BKG = 0.005  # Background fraction


def compute_efficiency_from_eq5(eff_trig, eff_reco):
    """
    Equation (5) from ATLAS note:
    ε_Z = (1 - (1 - ε_trig)²) × ε_reco²
    
    This considers:
    - At least one of two leptons triggers (1 - (1-ε)²)
    - Both leptons pass reconstruction (ε²)
    """
    eff_trig_event = 1 - (1 - eff_trig)**2  # At least one triggers
    eff_reco_event = eff_reco**2  # Both reconstructed
    return eff_trig_event * eff_reco_event


def derive_luminosity_formula_components(df):
    """
    From the ATLAS note, we have:
        L = N × (1 - f_bkg) / (σ × A × ε × F × t)
    
    And the data contains:
        ZeeLumi = L (instantaneous luminosity)
        ZeeRaw = N
        ZeeEffComb = ε (event-level efficiency)
        ZeeEffAComb = A × ε (acceptance × efficiency)
    
    So if we invert:
        σ × A × F × t = N × (1 - f_bkg) / (L × ε)
    
    Since we're computing instantaneous lumi (not per-second), 
    the formula simplifies.
    """
    
    # First, verify equation (5) matches ZeeEffComb
    print("=" * 60)
    print("STEP 1: VERIFY EQUATION (5) - Combined Efficiency")
    print("=" * 60)
    print("Formula: ε_Z = (1 - (1 - ε_trig)²) × ε_reco²")
    print()
    
    eff_calc = compute_efficiency_from_eq5(df['ZeeEffTrig'], df['ZeeEffReco'])
    eff_diff = eff_calc - df['ZeeEffComb']
    
    print(f"Calculated ε (eq.5) vs ZeeEffComb:")
    print(f"  Mean difference: {eff_diff.mean():.6f}")
    print(f"  Std difference:  {eff_diff.std():.6f}")
    print(f"  Max |diff|:      {eff_diff.abs().max():.6f}")
    
    # Show sample values
    print("\nSample comparison (first 5 rows):")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        calc = compute_efficiency_from_eq5(row['ZeeEffTrig'], row['ZeeEffReco'])
        print(f"  Row {i}: Calculated={calc:.6f}, ZeeEffComb={row['ZeeEffComb']:.6f}, "
              f"Diff={calc - row['ZeeEffComb']:+.6f}")
    
    print()
    return eff_calc


def derive_acceptance_and_mc_factor(df, eff_calc):
    """
    From the data:
        ZeeEffAComb = A_MC × ε_T&P × F_MC
    
    So:
        A × F = ZeeEffAComb / ε_T&P
    
    Also, the luminosity formula gives us:
        ZeeLumi [nb^-1] = ZeeRaw × (1 - f_bkg) / (σ × A × ε × F)
        
    Rearranging:
        σ × A × F = ZeeRaw × (1 - f_bkg) / (ZeeLumi × ε)
    """
    print("=" * 60)
    print("STEP 2: DERIVE ACCEPTANCE × MC FACTOR")
    print("=" * 60)
    
    # Method 1: From ZeeEffAComb / ZeeEffComb
    A_x_F_from_eff = df['ZeeEffAComb'] / df['ZeeEffComb']
    print(f"Method 1: A×F = ZeeEffAComb / ZeeEffComb")
    print(f"  Mean: {A_x_F_from_eff.mean():.6f}")
    print(f"  Std:  {A_x_F_from_eff.std():.6f}")
    
    # Method 2: From luminosity formula
    # σ × A × F = N × (1 - f_bkg) / (L × ε)
    # A × F = N × (1 - f_bkg) / (L × ε × σ)
    sigma_pb = SIGMA_THEORY  # pb
    sigma_nb = sigma_pb / 1000  # convert to nb (1 nb = 1000 pb)
    
    # ZeeLumi is in nb^-1, so we need σ in nb
    A_x_F_from_lumi = (df['ZeeRaw'] * (1 - F_BKG)) / (df['ZeeLumi'] * df['ZeeEffComb'] * sigma_nb)
    
    print(f"\nMethod 2: From luminosity formula with σ = {SIGMA_THEORY} pb = {sigma_nb} nb")
    print(f"  A×F = N × (1-f_bkg) / (L × ε × σ)")
    print(f"  Mean: {A_x_F_from_lumi.mean():.6f}")
    print(f"  Std:  {A_x_F_from_lumi.std():.6f}")
    
    print()
    return A_x_F_from_eff.mean(), A_x_F_from_lumi.mean()


def validate_full_formula(df, A_x_F):
    """
    Validate the full luminosity formula:
        ZeeLumi = ZeeRaw × (1 - f_bkg) / (σ × A × ε × F)
    """
    print("=" * 60)
    print("STEP 3: VALIDATE FULL LUMINOSITY FORMULA")
    print("=" * 60)
    
    sigma_nb = SIGMA_THEORY / 1000  # Convert pb to nb
    
    # Compute ZeeLumi using the formula
    # Using ZeeEffAComb which already includes A × ε × F
    zl_derived = (df['ZeeRaw'] * (1 - F_BKG)) / (sigma_nb * df['ZeeEffAComb'])
    
    residual = (zl_derived - df['ZeeLumi']) / df['ZeeLumi']
    
    print(f"Formula: ZeeLumi = ZeeRaw × (1 - f_bkg) / (σ × ZeeEffAComb)")
    print(f"         with σ = {SIGMA_THEORY} pb = {sigma_nb:.3f} nb")
    print()
    print(f"Residual (derived - actual) / actual:")
    print(f"  Mean:  {residual.mean()*100:+.4f}%")
    print(f"  Std:   {residual.std()*100:.4f}%")
    print(f"  Max |diff|: {residual.abs().max()*100:.4f}%")
    
    print("\nSample validation (first 5 rows):")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        derived = (row['ZeeRaw'] * (1 - F_BKG)) / (sigma_nb * row['ZeeEffAComb'])
        pct = (derived - row['ZeeLumi']) / row['ZeeLumi'] * 100
        print(f"  Row {i}: ZeeRaw={row['ZeeRaw']:.0f}, EffAComb={row['ZeeEffAComb']:.4f}")
        print(f"          Derived={derived:.4f}, Actual={row['ZeeLumi']:.4f}, Diff={pct:+.2f}%")
    
    return residual


def main():
    print("=" * 60)
    print("ATLAS Z-COUNTING FORMULA VALIDATION")
    print("Using ATL-DAPR-PUB-2021-001 equations")
    print("=" * 60)
    print()
    
    print("Loading Set 3 data (2018)...")
    df = pd.read_csv(DATA_PATH, nrows=500)
    print(f"Loaded {len(df)} rows")
    
    # Filter valid rows
    df = df[(df['ZeeRaw'] > 0) & (df['ZeeEffAComb'] > 0)].copy()
    print(f"Valid rows: {len(df)}")
    print()
    
    # Step 1: Verify equation (5)
    eff_calc = derive_luminosity_formula_components(df)
    
    # Step 2: Derive A × F
    A_F_eff, A_F_lumi = derive_acceptance_and_mc_factor(df, eff_calc)
    
    # Step 3: Validate full formula
    residual = validate_full_formula(df, A_F_eff)
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("The ATLAS Z-counting formula is:")
    print()
    print("  L_Z = N_Z × (1 - f_bkg) / (σ × A × ε × F)")
    print()
    print("Where in Set3 data:")
    print(f"  - N_Z = ZeeRaw (raw Z count)")
    print(f"  - f_bkg = {F_BKG} (background fraction)")
    print(f"  - σ = {SIGMA_THEORY} pb = 1.970 nb (theory cross-section)")
    print(f"  - A × ε × F = ZeeEffAComb (acceptance × eff × MC correction)")
    print()
    print("This simplifies to:")
    print("  ZeeLumi = ZeeRaw × 0.995 / (1.970 × ZeeEffAComb)")
    print()
    
    # Final validation
    mean_resid = residual.mean() * 100
    std_resid = residual.std() * 100
    if abs(mean_resid) < 0.1 and std_resid < 1.0:
        print("✓ VALIDATION PASSED: Formula matches within statistical uncertainty")
    else:
        print(f"⚠ VALIDATION: Mean residual = {mean_resid:.4f}%, Std = {std_resid:.4f}%")


if __name__ == "__main__":
    main()
