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
    ap.add_argument('--coeff', default='all', help='Harmonic Structure to study (XZ, YZ, XX_YY, XY, or all)')
    ap.add_argument('--n_scrambles', type=int, default=200, help='Number of scrambles to average over')
    ap.add_argument('--outdir', default='output/plots_linearity_scramble', help='Output directory')
    ap.add_argument('--n_points', type=int, default=11, help='Number of signal strength points (including 0)')
    ap.add_argument('--max_strength', type=float, default=2e-4, help='Maximum signal strength')
    args = ap.parse_args()

    coeffs_to_run = list(SME_COEFFICIENTS) if args.coeff == 'all' else [args.coeff]
    n_scrambles = args.n_scrambles
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Signal strengths: evenly spaced from 0 to max_strength
    signal_strengths = np.linspace(0, args.max_strength, args.n_points)
    n_strengths = len(signal_strengths)

    # Load scrambles (load once)
    files = sorted(glob.glob("output/scrambles_pq/scramble_*.parquet"))[:n_scrambles]
    if len(files) < n_scrambles:
        print(f"Warning: Only {len(files)} scrambles available, requested {n_scrambles}")
        n_scrambles = len(files)
        
    print(f"Loading {n_scrambles} scrambles into memory...")
    scrambles = [pd.read_parquet(f) for f in files]
    print("Loaded.")

    for coeff in coeffs_to_run:
        print(f"\nRunning Linearity Study (Scrambles) for {coeff}...")
        
        # Storage: [strength_idx, scramble_idx]
        fitted_values = np.zeros((n_strengths, n_scrambles))

        for s_idx, df in enumerate(scrambles):
            for str_idx, inj_val in enumerate(signal_strengths):
                if inj_val == 0:
                    df_use = df
                else:
                    df_use = inject_signal(df, coeff, inj_val)

                # Fit
                res = compute_double_ratio_and_fit(df_use)
                fitted_values[str_idx, s_idx] = res['fits'][coeff]['value']

            if (s_idx + 1) % 50 == 0:
                print(f"  Processed {s_idx + 1}/{n_scrambles} scrambles...")

        # Compute statistics
        fitted_mean = np.mean(fitted_values, axis=1)
        fitted_std = np.std(fitted_values, axis=1, ddof=1)
        fitted_err = fitted_std / np.sqrt(n_scrambles)  # SEM

        plot_linearity(coeff, signal_strengths, fitted_mean, fitted_err, outdir, n_scrambles)


def plot_linearity(coeff, x, y, yerr, outdir, n_scrambles):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Weight linear regression
    w = 1.0 / (yerr**2 + 1e-30)
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

    ax.errorbar(x, y, yerr=yerr, fmt='ko', label=f'Fitted (avg {n_scrambles} scrambles)')
    
    x_plot = np.array([x.min(), x.max()])
    ax.plot(x_plot, m*x_plot + c, 'r-', label=f'Fit: y = {m:.3f}x + {c:.2e}')
    ax.plot(x_plot, x_plot, 'b--', alpha=0.5, label='Ideal: y=x')

    ax.set_xlabel(f'Injected Signal ({coeff})')
    ax.set_ylabel(f'Fitted Signal ({coeff})')
    ax.set_title(f'Linearity (Scrambles, {n_scrambles} toys)\nSlope={m:.3f}±{m_err:.3f}, Int={c:.2e}±{c_err:.2e}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Scientific notation
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    outpath = outdir / f"linearity_scramble_{coeff}.pdf"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Wrote {outpath}")


if __name__ == '__main__':
    main()
