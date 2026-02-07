#!/usr/bin/env python
"""
plot_pulls.py

Analyze 1000 scramble results:
1. Plot pull distributions (value/error) for all coefficients
2. Identify outlier scrambles (>3σ significance)
3. Generate detailed plots for top outliers
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

from sme_fit import SME_COEFFICIENTS
from run_double_ratio import make_plot, phase_binned_double_ratio

def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def plot_pulls_and_pvalues(results: dict, outdir: str):
    """
    Plot pull distributions and calculate p-values.
    Pull = value / error (assuming null hypothesis c=0)
    """
    n_scrambles = len(results['scrambles'])
    
    # Extract pulls
    pulls = {c: [] for c in SME_COEFFICIENTS}
    for s in results['scrambles']:
        for c in SME_COEFFICIENTS:
            fit = s['fits'][c]
            if fit['error'] > 0:
                pulls[c].append(fit['value'] / fit['error'])
    
    # 1. Pull Distributions Plot
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
    
    x = np.linspace(-5, 5, 200)
    gaussian = stats.norm.pdf(x, 0, 1)
    
    outliers = {}
    
    for i, coeff in enumerate(SME_COEFFICIENTS):
        ax = axes[i]
        vals = np.array(pulls[coeff])
        
        # Identify outliers (>3 sigma)
        mask = np.abs(vals) > 3.0
        if mask.any():
            outliers[coeff] = np.where(mask)[0].tolist()
        
        # Plot histogram
        n, bins, patches = ax.hist(vals, bins=30, range=(-5, 5), density=True, 
                                  alpha=0.6, color='skyblue', edgecolor='black', linewidth=0.5)
        
        # Fit Gaussian
        mu, std = stats.norm.fit(vals)
        
        # Plot Standard Normal (Red) and Fit (Green)
        ax.plot(x, gaussian, 'r--', linewidth=1.5, label='N(0,1)')
        ax.plot(x, stats.norm.pdf(x, mu, std), 'g-', linewidth=2, alpha=0.7, 
               label=f'Fit: $\mu$={mu:.2f}, $\sigma$={std:.2f}')
        
        # Chi-square test for normality
        res = stats.normaltest(vals)
        
        ax.set_title(f"{coeff}")
        ax.text(0.05, 0.95, f"p(norm)={res.pvalue:.2f}", transform=ax.transAxes, 
                va='top', fontsize=8)
        
        if i == 0:
            ax.legend(fontsize=8)
            
    fig.suptitle(f"Pull Distributions (value/error) for {n_scrambles} Scrambles", fontsize=14)
    fig.tight_layout()
    outpath = os.path.join(outdir, "pull_distributions.pdf")
    fig.savefig(outpath)
    print(f"Wrote {outpath}")
    
    return outliers

def analyze_outliers(outliers: dict, scrambles_dir: str, results: dict, outdir: str):
    """
    Generate detailed plots for the most significant outliers.
    Similar to Sample #844 in the notebook.
    """
    if not os.path.exists(scrambles_dir):
        print("Scrambles directory not found, skipping detailed plots.")
        return

    print("\n" + "="*50)
    print("OUTLIER ANALYSIS (>3σ)")
    print("="*50)
    
    for coeff, idxs in outliers.items():
        for idx in idxs:
            # Get details
            scramble = results['scrambles'][idx]
            fit = scramble['fits'][coeff]
            sig = fit['value'] / fit['error']
            
            print(f"Scramble #{idx}: {coeff} = {fit['value']:.2e} +/- {fit['error']:.2e} ({sig:.2f}σ)")
            
            # Load the Parquet file
            pq_path = os.path.join(scrambles_dir, f"scramble_{idx:04d}.parquet")
            if os.path.exists(pq_path):
                df = pd.read_parquet(pq_path)
                
                # Make the plot
                top_text =f"Scramble #{idx}: {coeff} signal {sig:.2f}$\sigma$"
                
                # Compute DR
                x, y, yerr, info = phase_binned_double_ratio(
                    df, 'ZllLumi', 'ZllLumiErr', 'OffLumi', 'sday', 840.0
                )
                
                series = [{
                    'x': x, 'y': y, 'yerr': yerr, 
                    'fmt': 'o', 'label': f'{coeff} outlier'
                }]
                
                plot_path = os.path.join(outdir, f"outlier_{coeff}_scr{idx}.pdf")
                make_plot(plot_path, f"Outlier: {coeff} ({sig:.2f}$\sigma$)", top_text, "", series)
                print(f"  -> Wrote plot: {plot_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='Path to batch_1000_results.json')
    ap.add_argument('--scrambles_dir', default='output/scrambles_pq', help='Directory with Parquet files')
    ap.add_argument('--outdir', default='output/plots_analysis', help='Output directory')
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    # 1. Pulls
    outliers = plot_pulls_and_pvalues(results, args.outdir)
    
    # 2. Outlier Analysis
    analyze_outliers(outliers, args.scrambles_dir, results, args.outdir)

if __name__ == '__main__':
    main()
