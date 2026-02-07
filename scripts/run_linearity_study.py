#!/usr/bin/env python
"""
run_linearity_study.py

Linearity Study for SME Signal Injection:
1. Inject signals of various strengths (including 0) into scrambles.
2. Fit to recover the signal strength.
3. Plot: Injected (x-axis) vs Fitted (y-axis).
4. Perform three fit cases:
   - Fix slope=1, fit intercept only (check for spurious signal)
   - Fix intercept=0, fit slope only (check for sensitivity)
   - Float both (full consistency check)

Usage:
    python run_linearity_study.py --coeff duYZ --n_scrambles 100
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
    ap.add_argument('--coeff', default='duYZ', help='SME coefficient to study')
    ap.add_argument('--n_scrambles', type=int, default=100, help='Number of scrambles to average over')
    ap.add_argument('--outdir', default='output/plots_analysis', help='Output directory')
    ap.add_argument('--n_points', type=int, default=11, help='Number of signal strength points (including 0)')
    ap.add_argument('--max_strength', type=float, default=1e-4, help='Maximum signal strength')
    args = ap.parse_args()

    coeff = args.coeff
    n_scrambles = args.n_scrambles
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # Signal strengths: evenly spaced from 0 to max_strength
    signal_strengths = np.linspace(0, args.max_strength, args.n_points)
    n_strengths = len(signal_strengths)

    print(f"Linearity Study: coeff={coeff}, n_scrambles={n_scrambles}")
    print(f"Signal strengths (linear): {signal_strengths}")

    # Load scrambles
    files = sorted(glob.glob("output/scrambles_pq/scramble_*.parquet"))[:n_scrambles]
    if len(files) < n_scrambles:
        print(f"Warning: Only {len(files)} scrambles available, requested {n_scrambles}")
        n_scrambles = len(files)

    # Storage: [strength_idx, scramble_idx]
    fitted_values = np.zeros((n_strengths, n_scrambles))

    for s_idx, fpath in enumerate(files):
        df = pd.read_parquet(fpath)

        for str_idx, inj_val in enumerate(signal_strengths):
            if inj_val == 0:
                # No injection
                df_use = df
            else:
                # Inject signal
                df_use = inject_signal(df, coeff, inj_val)

            # Fit
            res = compute_double_ratio_and_fit(df_use)
            fitted_values[str_idx, s_idx] = res['fits'][coeff]['value']

        if (s_idx + 1) % 10 == 0:
            print(f"  Processed {s_idx + 1}/{n_scrambles} scrambles...")

    # Compute statistics
    fitted_mean = np.mean(fitted_values, axis=1)
    fitted_std = np.std(fitted_values, axis=1, ddof=1)
    fitted_err = fitted_std / np.sqrt(n_scrambles)  # SEM

    print("\nResults:")
    for i, inj in enumerate(signal_strengths):
        print(f"  Injected={inj:.2e} -> Fitted={fitted_mean[i]:.2e} ± {fitted_err[i]:.2e}")

    X = signal_strengths
    Y = fitted_mean
    w = 1.0 / (fitted_err**2 + 1e-30)  # Weights

    # ========== FIT CASE 1: Fix slope=1, fit intercept ==========
    # Model: Y = 1*X + b => Y - X = b
    # Weighted mean of (Y - X)
    residuals = Y - X
    intercept_fixed_slope = np.sum(w * residuals) / np.sum(w)
    intercept_fixed_slope_err = 1.0 / np.sqrt(np.sum(w))
    chi2_fixed_slope = np.sum(w * (residuals - intercept_fixed_slope)**2)
    ndof_fixed_slope = len(X) - 1

    # ========== FIT CASE 2: Fix intercept=0, fit slope ==========
    # Model: Y = m*X => Y/X = m (weighted)
    # Weighted: sum(w*X*Y) / sum(w*X*X)
    slope_fixed_int = np.sum(w * X * Y) / np.sum(w * X * X)
    slope_fixed_int_err = 1.0 / np.sqrt(np.sum(w * X * X))
    chi2_fixed_int = np.sum(w * (Y - slope_fixed_int * X)**2)
    ndof_fixed_int = len(X) - 1

    # ========== FIT CASE 3: Float both slope and intercept ==========
    sum_w = np.sum(w)
    sum_wx = np.sum(w * X)
    sum_wy = np.sum(w * Y)
    sum_wxx = np.sum(w * X * X)
    sum_wxy = np.sum(w * X * Y)

    denom = sum_w * sum_wxx - sum_wx**2
    slope_float = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
    intercept_float = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denom
    slope_float_err = np.sqrt(sum_w / denom)
    intercept_float_err = np.sqrt(sum_wxx / denom)
    chi2_float = np.sum(w * (Y - slope_float * X - intercept_float)**2)
    ndof_float = len(X) - 2

    # Print results
    print("\n" + "="*70)
    print("FIT RESULTS:")
    print("="*70)
    print(f"\nCase 1: Fix slope=1, fit intercept")
    print(f"  Intercept = {intercept_fixed_slope:.2e} ± {intercept_fixed_slope_err:.2e}")
    print(f"  Chi2/ndof = {chi2_fixed_slope:.2f}/{ndof_fixed_slope} = {chi2_fixed_slope/ndof_fixed_slope:.2f}")

    print(f"\nCase 2: Fix intercept=0, fit slope")
    print(f"  Slope = {slope_fixed_int:.4f} ± {slope_fixed_int_err:.4f}")
    print(f"  Chi2/ndof = {chi2_fixed_int:.2f}/{ndof_fixed_int} = {chi2_fixed_int/ndof_fixed_int:.2f}")

    print(f"\nCase 3: Float both")
    print(f"  Slope = {slope_float:.4f} ± {slope_float_err:.4f}")
    print(f"  Intercept = {intercept_float:.2e} ± {intercept_float_err:.2e}")
    print(f"  Chi2/ndof = {chi2_float:.2f}/{ndof_float} = {chi2_float/ndof_float:.2f}")
    print("="*70)

    # ========== PLOT 1: Intercept Study (Fix Slope=1) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.errorbar(X, Y, yerr=fitted_err, 
                fmt='o', color='black', markersize=6, capsize=3,
                label='Fitted (avg over scrambles)')

    x_fit = np.linspace(0, args.max_strength, 100)
    y_case1 = x_fit + intercept_fixed_slope
    ax.plot(x_fit, y_case1, 'r-', linewidth=2, 
            label=f'Fit: intercept = {intercept_fixed_slope:.2e} ± {intercept_fixed_slope_err:.2e}')
    ax.plot(x_fit, x_fit, 'b--', linewidth=1, alpha=0.5, label='Ideal (intercept = 0)')

    ax.set_xlabel('Injected Signal Strength', fontsize=12)
    ax.set_ylabel('Fitted Signal Strength', fontsize=12)
    ax.set_title(f'Intercept Study (Fix Slope=1): {coeff}\n(Averaged over {n_scrambles} scrambles, χ²/ndf = {chi2_fixed_slope:.1f}/{ndof_fixed_slope})', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-args.max_strength * 0.05, args.max_strength * 1.05)
    ax.set_ylim(min(Y.min(), -args.max_strength * 0.1), args.max_strength * 1.1)

    outpath1 = outdir / f"intercept_study_{coeff}.pdf"
    plt.tight_layout()
    plt.savefig(outpath1, dpi=150)
    plt.close()
    print(f"\nWrote {outpath1}")

    # ========== PLOT 2: Slope Study (Fix Intercept=0) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.errorbar(X, Y, yerr=fitted_err, 
                fmt='o', color='black', markersize=6, capsize=3,
                label='Fitted (avg over scrambles)')

    y_case2 = slope_fixed_int * x_fit
    ax.plot(x_fit, y_case2, 'r-', linewidth=2, 
            label=f'Fit: slope = {slope_fixed_int:.4f} ± {slope_fixed_int_err:.4f}')
    ax.plot(x_fit, x_fit, 'b--', linewidth=1, alpha=0.5, label='Ideal (slope = 1)')

    ax.set_xlabel('Injected Signal Strength', fontsize=12)
    ax.set_ylabel('Fitted Signal Strength', fontsize=12)
    ax.set_title(f'Slope Study (Fix Intercept=0): {coeff}\n(Averaged over {n_scrambles} scrambles, χ²/ndf = {chi2_fixed_int:.1f}/{ndof_fixed_int})', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-args.max_strength * 0.05, args.max_strength * 1.05)
    ax.set_ylim(min(Y.min(), -args.max_strength * 0.1), args.max_strength * 1.1)

    outpath2 = outdir / f"slope_study_{coeff}.pdf"
    plt.tight_layout()
    plt.savefig(outpath2, dpi=150)
    plt.close()
    print(f"Wrote {outpath2}")

    # ========== PLOT 3: Linearity Study (Both Float) ==========
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.errorbar(X, Y, yerr=fitted_err, 
                fmt='o', color='black', markersize=6, capsize=3,
                label='Fitted (avg over scrambles)')

    y_case3 = slope_float * x_fit + intercept_float
    ax.plot(x_fit, y_case3, 'r-', linewidth=2, 
            label=f'Fit: slope = {slope_float:.4f} ± {slope_float_err:.4f}, int = {intercept_float:.2e} ± {intercept_float_err:.2e}')
    ax.plot(x_fit, x_fit, 'b--', linewidth=1, alpha=0.5, label='Ideal (y = x)')

    ax.set_xlabel('Injected Signal Strength', fontsize=12)
    ax.set_ylabel('Fitted Signal Strength', fontsize=12)
    ax.set_title(f'Linearity Study (Both Float): {coeff}\n(Averaged over {n_scrambles} scrambles, χ²/ndf = {chi2_float:.1f}/{ndof_float})', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-args.max_strength * 0.05, args.max_strength * 1.05)
    ax.set_ylim(min(Y.min(), -args.max_strength * 0.1), args.max_strength * 1.1)

    outpath3 = outdir / f"linearity_study_{coeff}.pdf"
    plt.tight_layout()
    plt.savefig(outpath3, dpi=150)
    plt.close()
    print(f"Wrote {outpath3}")

    # Summary interpretation
    print("\n" + "="*70)
    print("INTERPRETATION:")
    
    # Case 1 interpretation
    if abs(intercept_fixed_slope) < 3 * intercept_fixed_slope_err:
        print("  Case 1: ✓ Intercept consistent with 0 (no spurious signal)")
    else:
        print(f"  Case 1: ⚠ Intercept = {intercept_fixed_slope:.1e} deviates from 0")

    # Case 2 interpretation
    if abs(slope_fixed_int - 1.0) < 3 * slope_fixed_int_err:
        print("  Case 2: ✓ Slope consistent with 1.0 (correct sensitivity)")
    else:
        print(f"  Case 2: ⚠ Slope = {slope_fixed_int:.4f} deviates from 1.0")

    # Case 3 interpretation
    if abs(slope_float - 1.0) < 3 * slope_float_err and abs(intercept_float) < 3 * intercept_float_err:
        print("  Case 3: ✓ Both slope and intercept consistent with ideal")
    else:
        print("  Case 3: ⚠ Check individual parameters above")

    print("="*70)


if __name__ == '__main__':
    main()
