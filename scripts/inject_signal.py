#!/usr/bin/env python
"""
inject_signal.py

Injects artificial SME signals into a dataset (scrambled or original) 
and verifies if the fitting pipeline recovers them.

Methodology:
1. Load dataset (Parquet)
2. Calculate phase for each event
3. Modulate ZllLumi: Z_new = Z_old * (1 + c * f(phi))
   where f(phi) is the SME template function.
4. Run standard double ratio & fitting analysis.
5. Report Injected vs Fitted values.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add scripts dir to path to import local modules
sys.path.append(str(Path(__file__).parent))

from sme_fit import sme_template, SME_COEFFICIENTS
from run_batch_analysis import compute_double_ratio_and_fit, fold_phase, SIDEREAL_DAY_H, FIXED_COLS

def inject_signal(df: pd.DataFrame, injection_coeff: str, injection_value: float) -> pd.DataFrame:
    """
    Inject SME signal into the dataframe.
    Modulates 'ZllLumi' column.
    """
    df_mod = df.copy()
    
    # Calculate phase
    tmid = 0.5 * (df['LBStart'].to_numpy() + df['LBEnd'].to_numpy())
    phi = fold_phase(tmid, SIDEREAL_DAY_H)
    
    # Get modulation function
    # RD(phi) = 1 + c * f(phi)
    # Since RD = N/D, and we assume D is unaffected, we modulate N.
    # N_new = N_old * (1 + c * f(phi))
    
    modulation = 1.0 + injection_value * sme_template(injection_coeff, phi)
    
    # Apply modulation
    df_mod['ZllLumi'] *= modulation
    # Also scale error? Yes, relative error scaling.
    df_mod['ZllLumiErr'] *= modulation
    
    return df_mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', required=True, help='Input Parquet file (null/scrambled)')
    ap.add_argument('--coeff', required=True, choices=SME_COEFFICIENTS, help='Coefficient to inject')
    ap.add_argument('--value', type=float, required=True, help='Value to inject')
    ap.add_argument('--bin_sec', type=float, default=840.0, help='Phase bin size')
    args = ap.parse_args()
    
    print(f"Loading {args.infile} ...")
    df = pd.read_parquet(args.infile)
    print(f"Loaded {len(df)} rows.")
    
    print(f"Injecting {args.coeff} = {args.value:.2e} ...")
    df_injected = inject_signal(df, args.coeff, args.value)
    
    print("Running standard analysis on injected data...")
    result = compute_double_ratio_and_fit(df_injected, bin_sec=args.bin_sec)
    
    fits = result['fits']
    
    print("\n" + "="*60)
    print(f"INJECTION RESULTS: {args.coeff}")
    print("="*60)
    
    # Check the injected coefficient
    fit = fits[args.coeff]
    val = fit['value']
    err = fit['error']
    sig = fit['significance']
    
    pull = (val - args.value) / err
    bias_pct = 100.0 * (val - args.value) / args.value if args.value != 0 else 0
    
    print(f"Injected: {args.value:.4e}")
    print(f"Fitted:   {val:.4e} +/- {err:.4e}")
    print(f"Bias:     {val - args.value:.4e} ({bias_pct:.2f}%)")
    print(f"Pull:     {pull:.2f} σ")
    
    print("-" * 60)
    print("Cross Check (Top 3 other signals):")
    # Sort remaining by significance
    others = [(k, v) for k, v in fits.items() if k != args.coeff]
    others.sort(key=lambda x: x[1]['significance'], reverse=True)
    
    for k, v in others[:3]:
        print(f"{k}: {v['value']:.2e} +/- {v['error']:.2e} ({v['significance']:.2f}σ)")
    
    print("="*60)

if __name__ == '__main__':
    main()
