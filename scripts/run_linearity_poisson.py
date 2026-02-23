#!/usr/bin/env python
"""
run_linearity_poisson.py

Re-implementation of linearity study using Poisson-fluctuated toys (Method B)
instead of Scrambles. Works with the new 4-harmonic geometric basis.

Methodology:
1. Load original data (set3_pruned.parquet).
2. For each toy:
   a. Generate Background:
      - Calculate Effective Counts: Neff = (Lumi/Err)**2
      - Fluctuate: N_toy ~ Poisson(Neff)
      - Scale back: Lumi_toy = N_toy * (Lumi/Neff)
      - Error_toy = Err * sqrt(N_toy/Neff)
   b. Inject Signal:
      - Scale Lumi_toy by (1 + Signal * Template(phi))
   c. Fit:
      - Standard Double Ratio fit.
3. Plot Fitted vs Injected.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import glob

# Add scripts dir to path
sys.path.append(str(Path(__file__).parent))

from sme_fit import SME_COEFFICIENTS, sme_template
from inject_signal import inject_signal
from run_batch_analysis import compute_double_ratio_and_fit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coeff', default='all', choices=SME_COEFFICIENTS + ('all',), 
                   help='Harmonic Structure to study (XZ, YZ, XX_YY, XY, or all)')
    ap.add_argument('--input_file', default='input/set3_pruned.parquet', help='Path to input parquet file')
    ap.add_argument('--n_toys', type=int, default=200, help='Number of Poisson toys')
    ap.add_argument('--outdir', default='output/plots_linearity_poisson', help='Output directory')
    ap.add_argument('--n_points', type=int, default=6, help='Number of signal strength points')
    ap.add_argument('--max_strength', type=float, default=2e-4, help='Maximum signal strength')
    args = ap.parse_args()

    coeffs_to_run = list(SME_COEFFICIENTS) if args.coeff == 'all' else [args.coeff]
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    print(f"Loading data from {args.input_file}...")
    df_orig = pd.read_parquet(args.input_file)
    print(f"Loaded {len(df_orig)} LBs.")
    
    # Determine suffix for plots
    input_name = Path(args.input_file).stem
    suffix = ""
    if "scramble" in input_name:
        # Extract number? or just use name
        suffix = f"_{input_name}"

    
    # Calculate Effective Counts Parameters once
    # Neff = (Val/Err)^2
    # S = Val/Neff = Err^2 / Val
    
    # Filter valid
    mask = (df_orig['ZllLumi'] > 0) & (df_orig['ZllLumiErr'] > 0)
    df_valid = df_orig[mask].copy()
    
    val = df_valid['ZllLumi'].to_numpy()
    err = df_valid['ZllLumiErr'].to_numpy()
    
    neff = (val / err)**2
    scale = val / neff
    
    print(f"Effective stats check: Mean Neff = {np.mean(neff):.1f}")
    
    signal_strengths = np.linspace(0, args.max_strength, args.n_points)
    
    for coeff in coeffs_to_run:
        print(f"\nrunning Linearity Study (Poisson Neff) for {coeff}...")
        
        results_mean = np.zeros(len(signal_strengths))
        results_err = np.zeros(len(signal_strengths))
        
        for i_s, strength in enumerate(signal_strengths):
            fits = []
            print(f"  Signal {strength:.2e} ({i_s+1}/{len(signal_strengths)}): ", end='', flush=True)
            
            for i_toy in range(args.n_toys):
                if i_toy % 50 == 0: print(".", end='', flush=True)
                
                # 1. Generate Poisson Toy
                # N_toy ~ Poisson(Neff)
                n_toy = np.random.poisson(neff)
                
                # Reconstruct Toy Dataframe
                lumi_toy = n_toy * scale
                
                # Error scaling
                # Avoid div by zero
                valid_sc = neff > 0
                err_toy = np.zeros_like(err)
                err_toy[valid_sc] = err[valid_sc] * np.sqrt(n_toy[valid_sc] / neff[valid_sc])
                
                df_toy = df_valid.copy()
                df_toy['ZllLumi'] = lumi_toy
                df_toy['ZllLumiErr'] = err_toy
                
                # 2. Inject Signal
                if strength != 0:
                    df_toy = inject_signal(df_toy, coeff, strength)
                    
                # 3. Fit
                res = compute_double_ratio_and_fit(df_toy)
                fit_val = res['fits'][coeff]['value']
                fits.append(fit_val)
            
            print(" Done.")
            
            fits = np.array(fits)
            results_mean[i_s] = np.mean(fits)
            results_err[i_s] = np.std(fits) / np.sqrt(len(fits)) # Standard Error of Mean
            
        # Plotting
        plot_linearity(coeff, signal_strengths, results_mean, results_err, outdir, args.n_toys, suffix)

def plot_linearity(coeff, x, y, yerr, outdir, n_toys, suffix=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Fit line y = mx + c
    w = 1.0 / (yerr**2 + 1e-30)
    valid = np.isfinite(y) & np.isfinite(yerr)
    if valid.sum() < 2:
        return

    x = x[valid]
    y = y[valid]
    w = w[valid]
    
    sum_w = np.sum(w)
    sum_wx = np.sum(w*x)
    sum_wy = np.sum(w*y)
    sum_wxx = np.sum(w*x*x)
    sum_wxy = np.sum(w*x*y)
    
    denom = sum_w * sum_wxx - sum_wx**2
    m = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
    c = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denom
    
    m_err = np.sqrt(sum_w / denom)
    c_err = np.sqrt(sum_wxx / denom)
    
    ax.errorbar(x, y, yerr=yerr, fmt='ko', label='Fitted Mean')
    
    x_plot = np.array([x.min(), x.max()])
    ax.plot(x_plot, m*x_plot + c, 'r-', label=f'Fit: y = {m:.3f}x + {c:.2e}')
    ax.plot(x_plot, x_plot, 'b--', alpha=0.5, label='Ideal: y=x')
    
    ax.set_xlabel(f"Injected Signal ({coeff})")
    ax.set_ylabel(f"Fitted Signal ({coeff})")
    ax.set_title(f"Linearity (Poisson PEs, {n_toys} toys)\nSlope={m:.3f}±{m_err:.3f}, Int={c:.2e}±{c_err:.2e}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scientific notation
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    sname = f"linearity_poisson_{coeff}{suffix}.pdf"
    plt.savefig(outdir / sname)
    plt.close()
    print(f"Saved {outdir/sname}")

if __name__ == "__main__":
    main()
