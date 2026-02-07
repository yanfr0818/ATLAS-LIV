#!/usr/bin/env python
"""
plot_statistics.py

Generate statistical summary plots from batch analysis results.
Shows the distribution of SME coefficients across scrambles,
and compares with the original (unscrambled) data.

Usage:
    python plot_statistics.py --results_dir output/results --outdir output/plots
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sme_fit import SME_COEFFICIENTS


def load_batch_summary(results_dir: str) -> dict:
    """Load batch_summary.json from results directory."""
    path = Path(results_dir) / "batch_summary.json"
    with open(path) as f:
        return json.load(f)


def load_original_results(results_dir: str) -> dict:
    """Load original_results.json if exists."""
    path = Path(results_dir) / "original_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def extract_coefficient_arrays(summary: dict) -> dict:
    """Extract arrays of fitted values for each coefficient across all scrambles."""
    coeffs = {}
    for name in SME_COEFFICIENTS:
        values = []
        errors = []
        significances = []
        for scr in summary['scrambles']:
            fit = scr.get('fits', scr.get('sme_fits', {})).get(name)
            if fit:
                values.append(fit['value'])
                errors.append(fit['error'])
                significances.append(fit['significance'])
        coeffs[name] = {
            'values': np.array(values),
            'errors': np.array(errors),
            'significances': np.array(significances),
        }
    return coeffs


def plot_coefficient_distributions(
    scramble_coeffs: dict,
    original_results: dict,
    outpath: str,
    seed: int = None,
):
    """
    Plot histograms of coefficient values across scrambles,
    with the original data value marked.
    """
    n_coeffs = len(SME_COEFFICIENTS)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    
    for i, name in enumerate(SME_COEFFICIENTS):
        ax = axes[i]
        
        vals = scramble_coeffs[name]['values']
        
        # Histogram of scrambled values
        ax.hist(vals, bins=min(10, len(vals)), alpha=0.7, color='steelblue', 
                edgecolor='black', linewidth=0.5, label='Scrambles')
        
        # Mark original value if available
        if original_results:
            orig_val = original_results['sme_fits'][name]['value']
            ax.axvline(orig_val, color='red', linewidth=2, linestyle='--', 
                      label=f'Original: {orig_val:.2e}')
        
        # Statistics
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        
        ax.axvline(mean, color='green', linewidth=1.5, linestyle='-', alpha=0.8)
        ax.axvline(mean - std, color='green', linewidth=1, linestyle=':', alpha=0.6)
        ax.axvline(mean + std, color='green', linewidth=1, linestyle=':', alpha=0.6)
        
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Count')
        ax.set_title(name)
        
        # Add stats text
        stats_text = f'μ={mean:.1e}\nσ={std:.1e}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                va='top', ha='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title
    title = f'SME Coefficient Distributions ({len(scramble_coeffs["duXZ"]["values"])} scrambles)'
    if seed is not None:
        title += f', seed={seed}'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_significance_distributions(
    scramble_coeffs: dict,
    original_results: dict,
    outpath: str,
    seed: int = None,
):
    """
    Plot histograms of significance values (|value/error|) across scrambles.
    """
    n_coeffs = len(SME_COEFFICIENTS)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    
    for i, name in enumerate(SME_COEFFICIENTS):
        ax = axes[i]
        
        sigs = scramble_coeffs[name]['significances']
        
        # Histogram
        ax.hist(sigs, bins=min(10, len(sigs)), alpha=0.7, color='darkorange',
                edgecolor='black', linewidth=0.5, label='Scrambles')
        
        # Mark original significance if available
        if original_results:
            orig_sig = original_results['sme_fits'][name]['significance']
            ax.axvline(orig_sig, color='red', linewidth=2, linestyle='--',
                      label=f'Original: {orig_sig:.2f}σ')
        
        # Reference lines
        ax.axvline(1.0, color='gray', linewidth=1, linestyle=':', alpha=0.6)
        ax.axvline(2.0, color='gray', linewidth=1, linestyle=':', alpha=0.6)
        
        ax.set_xlabel('Significance (σ)')
        ax.set_ylabel('Count')
        ax.set_title(name)
        
        # Stats
        mean = np.mean(sigs)
        ax.text(0.95, 0.95, f'μ={mean:.2f}σ', transform=ax.transAxes,
                va='top', ha='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    title = f'SME Coefficient Significances ({len(scramble_coeffs["duXZ"]["significances"])} scrambles)'
    if seed is not None:
        title += f', seed={seed}'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_summary_table(
    scramble_coeffs: dict,
    original_results: dict,
    outpath: str,
):
    """
    Create a summary table comparing original vs scramble statistics.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Build table data
    headers = ['Coefficient', 'Original Value', 'Scramble Mean', 'Scramble Std', 
               'Original Signif', 'Scramble Mean Signif', '% > Original']
    
    rows = []
    for name in SME_COEFFICIENTS:
        vals = scramble_coeffs[name]['values']
        sigs = scramble_coeffs[name]['significances']
        
        scr_mean = np.mean(vals)
        scr_std = np.std(vals, ddof=1)
        scr_sig_mean = np.mean(sigs)
        
        if original_results:
            orig_val = original_results['sme_fits'][name]['value']
            orig_sig = original_results['sme_fits'][name]['significance']
            pct_greater = 100 * np.mean(np.abs(vals) > np.abs(orig_val))
        else:
            orig_val = np.nan
            orig_sig = np.nan
            pct_greater = np.nan
        
        rows.append([
            name,
            f'{orig_val:.2e}',
            f'{scr_mean:.2e}',
            f'{scr_std:.2e}',
            f'{orig_sig:.2f}σ',
            f'{scr_sig_mean:.2f}σ',
            f'{pct_greater:.0f}%',
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color header
    for j, cell in enumerate(table.get_celld().values()):
        if cell.get_text().get_text() in headers:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
    
    ax.set_title('SME Coefficient Comparison: Original vs Scrambles', 
                fontsize=12, fontweight='bold', pad=20)
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='output/results', help='Directory with JSON results')
    ap.add_argument('--outdir', default='output/plots_analysis', help='Output directory for plots')
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data
    summary = load_batch_summary(args.results_dir)
    original = load_original_results(args.results_dir)
    
    seed = summary.get('seed')
    n_scrambles = len(summary['scrambles'])
    print(f"Loaded {n_scrambles} scrambles, seed={seed}")
    
    if original:
        print("Loaded original results for comparison")
    
    # Extract coefficient arrays
    scramble_coeffs = extract_coefficient_arrays(summary)
    
    # Generate plots
    plot_coefficient_distributions(
        scramble_coeffs, original,
        os.path.join(args.outdir, 'sme_coefficient_distributions.pdf'),
        seed=seed,
    )
    
    plot_significance_distributions(
        scramble_coeffs, original,
        os.path.join(args.outdir, 'sme_significance_distributions.pdf'),
        seed=seed,
    )
    
    plot_summary_table(
        scramble_coeffs, original,
        os.path.join(args.outdir, 'sme_summary_table.pdf'),
    )
    
    print("\nDone! Generated statistical summary plots.")


if __name__ == '__main__':
    main()
