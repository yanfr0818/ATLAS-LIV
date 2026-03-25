#!/usr/bin/env python
"""
run_double_ratio.py

Compute double ratio from Set3 data and generate ATLAS-formatted plots.
Simplified design:
- Reads single combined Parquet/CSV (with Year separation)
- Uses direct phase binning with configurable bin duration per period
- Outputs 7 single-year ATLAS standardized plots per scramble/dataset

Usage:
    python run_double_ratio.py --infile output/scrambles_pq/scramble_0000.parquet --outdir output/plots_DR
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Period durations in hours
PER_H = {'sday': 24.0, 'day': 23.9344696, 'hour': 1.0}

def set_atlas_style():
    """Apply ATLAS-style matplotlib rcParams."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
        'font.size': 16,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'axes.linewidth': 1.5,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 8,
        'xtick.minor.size': 4,
        'ytick.major.size': 8,
        'ytick.minor.size': 4,
        'xtick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.major.width': 1.2,
        'ytick.minor.width': 0.8,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'legend.frameon': False,
        'legend.fontsize': 14,
        'figure.figsize': (8, 8),
        'figure.dpi': 150,
        'axes.grid': False,
    })


def add_atlas_label(ax, sqrts, lumi, label_type='Internal', x=0.05, y=0.95):
    """Add the 'ATLAS Internal' watermark and luminosity info."""
    ax.text(x, y, r'$\bf{ATLAS}$' + f' {label_type}',
            transform=ax.transAxes,
            fontsize=18, va='top', ha='left')
    ax.text(x, y - 0.055, fr'$\sqrt{{s}} = {sqrts}$ TeV, {lumi:.1f} fb$^{{-1}}$',
            transform=ax.transAxes,
            fontsize=14, va='top', ha='left')


def fold_phase(tmid: np.ndarray, per_h: float, t0: float = 953551209.0) -> np.ndarray:
    """Fold time into phase [0, 1) for a given period."""
    return np.mod((tmid - t0) / 3600.0, per_h) / per_h


def phase_binned_double_ratio(
    df: pd.DataFrame,
    num_col: str,
    numerr_col: str,
    den_col: str,
    per_tag: str,
    nbins: int = 100,
    t0: float = 953551209.0
) -> tuple:
    per_h = PER_H[per_tag]
    per_sec = per_h * 3600.0
    bin_sec = per_sec / nbins
    
    tmid = 0.5 * (df['LBStart'].to_numpy() + df['LBEnd'].to_numpy())
    phi = fold_phase(tmid, per_h, t0)
    
    idx = np.floor(phi * nbins).astype(int)
    idx = np.clip(idx, 0, nbins - 1)
    
    n = np.zeros(nbins, dtype=float)
    nerr2 = np.zeros(nbins, dtype=float)
    d = np.zeros(nbins, dtype=float)
    
    t_live = df['LBLive'].to_numpy(dtype=float)
    nv = df[num_col].to_numpy(dtype=float) * t_live
    ev = df[numerr_col].to_numpy(dtype=float) * t_live
    dv = df[den_col].to_numpy(dtype=float) * t_live
    
    np.add.at(n, idx, nv)
    np.add.at(nerr2, idx, ev * ev)
    np.add.at(d, idx, dv)
    
    ok = d > 0
    n = n[ok]
    nerr2 = nerr2[ok]
    d = d[ok]
    
    r = n / d
    r0 = n.sum() / d.sum()
    y = r / r0 - 1.0
    yerr = np.sqrt(nerr2) / (d * r0)
    
    nb_plot = len(y)
    x = (np.arange(nbins)[ok] + 0.5) / float(nbins)
    
    info = {
        'per_tag': per_tag,
        'per_h': per_h,
        'nbins': nbins,
        'nbins_filled': nb_plot,
        'bin_sec': bin_sec,
        'bin_min': bin_sec / 60.0,
    }
    return x, y, yerr, info


def make_atlas_dr_plot(outpath: str, x, y, yerr, per_tag, sqrts, lumi, info: dict, top_text: str):
    set_atlas_style()
    fig, ax = plt.subplots()
    
    color_map = {'sday': 'C1', 'day': 'C0', 'hour': 'C2'}
    fmt_map = {'sday': 's', 'day': 'o', 'hour': 'D'}
    label_map = {'sday': 'Solar Day', 'day': 'Sidereal Day', 'hour': '1 Hour'}
    
    color = color_map.get(per_tag, 'black')
    fmt = fmt_map.get(per_tag, 'o')
    per_label = label_map.get(per_tag, per_tag)
    
    ax.errorbar(x, y, yerr=yerr, fmt=fmt, color=color, markersize=5, 
                elinewidth=1.5, capsize=0, label=f'Data ({per_label})')
    
    ax.axhline(0.0, linewidth=1.5, color='red', linestyle='-', label='Ideal (DR-1 = 0)')
    
    mean_val = np.mean(y)
    rms_val = np.std(y)
    ax.axhline(mean_val, linewidth=1.2, color='blue', linestyle='--', label=f'Mean = {mean_val:.2e}')
    
    # RMS bands (dashed lines at ±RMS from mean)
    ax.axhline(mean_val + rms_val, linewidth=1.0, color='green', linestyle=':', label=f'±RMS = {rms_val:.2e}')
    ax.axhline(mean_val - rms_val, linewidth=1.0, color='green', linestyle=':')
    
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_xlabel('Phase', labelpad=10)
    ax.set_ylabel('Double Ratio - 1', labelpad=10)
    
    add_atlas_label(ax, sqrts, lumi, label_type='Internal')
    
    info_text = f"Bins: {info['nbins']} ({info['bin_min']:.1f} min)\n{top_text}"
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, va='top', ha='right', fontsize=12)
    
    ax.legend(loc='lower right')
    
    ylo = float(np.min(y - yerr))
    yhi = float(np.max(y + yerr))
    pad = 0.2 * (yhi - ylo) if yhi > ylo else 0.005
    min_span = 0.005
    if (yhi + pad) - (ylo - pad) < min_span:
        mid = (yhi + ylo) / 2
        ax.set_ylim(mid - min_span / 2, mid + min_span / 2)
    else:
        ax.set_ylim(ylo - pad, yhi + pad)
    
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Wrote {outpath}")


def run_single_dataset(args):
    """Process a single dataset to generate DR plots."""
    usecols = ['RunNum', 'LBStart', 'LBEnd', args.den, args.num, args.numerr, 'LBLive', 'Year']
    if args.infile.endswith('.parquet'):
        df = pd.read_parquet(args.infile, columns=usecols)
    else:
        df = pd.read_csv(args.infile, usecols=usecols)
        
    df = df[df[args.den] > 0]
    print(f"Loaded {len(df)} rows from {args.infile}")
    
    periods = [
        ('sday', 's', 'Solar Day'),
        ('day', 'o', 'Sidereal Day'),
    ]
    
    df['YearStr'] = df['Year'].astype(str)
    # Combine 15 and 16
    df.loc[df['Year'].isin([2015, 2016]), 'YearStr'] = '1516'
    
    # Combine 22 and 23
    df.loc[df['Year'].isin([2022, 2023]), 'YearStr'] = '2223'
    
    lumi_map = {
        '1516': 36.6,
        '2017': 44.6,
        '2018': 58.7,
        '2223': 51.1,
        '2024': 107.6,
        '2025': 114.7
    }
    
    sqrts_map = {
        '1516': 13,
        '2017': 13,
        '2018': 13,
        '2223': 13.6,
        '2024': 13.6,
        '2025': 13.6
    }

    
    year_groups = df.groupby('YearStr')
    
    # Process each year
    for yr, group in year_groups:
        if yr not in lumi_map:
            continue
            
        sqrts = sqrts_map[yr]
        lumi = lumi_map[yr]
        
        if args.scramble_idx is not None:
            top_text = f"Scramble: {args.scramble_idx:04d}"
        else:
            top_text = "Original Data"
        
        for per_tag, fmt, label in periods:
            x, y, yerr, info = phase_binned_double_ratio(
                group, args.num, args.numerr, args.den, per_tag, args.nbins, args.t0
            )
            
            scr_idx = args.name.split('_')[-1] if 'scramble' in args.name else 'data'
            outpath = os.path.join(args.outdir, f"DR_{scr_idx}_{yr}_{per_tag}.pdf")
            
            make_atlas_dr_plot(outpath, x, y, yerr, per_tag, sqrts, lumi, info, top_text)


def run_scrambles_batch(args):
    """Run batch analysis over multiple scrambles."""
    import json
    from pathlib import Path
    
    results_path = Path("output/results/batch_1000_results.json")
    scrambles_dir = Path("output/scrambles_pq")
    
    if not results_path.exists():
        print(f"Error: {results_path} not found!")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    master_seed = data["seed"]
    scrambles = data["scrambles"]
    
    print(f"Running batch processing for {args.n_scrambles} scrambles...")
    
    for i in range(args.n_scrambles):
        if i >= len(scrambles):
            break
            
        s = scrambles[i]
        idx = s["scramble_idx"]
        infile = scrambles_dir / f"scramble_{idx:04d}.parquet"
        
        if not infile.exists():
            print(f"Warning: {infile} not found, skipping...")
            continue
            
        import copy
        iter_args = copy.copy(args)
        iter_args.infile = str(infile)
        iter_args.name = f"scramble_{idx:04d}"
        iter_args.seed = master_seed
        iter_args.child_seed = s["child_seed"]
        iter_args.scramble_idx = idx
        
        print(f"\nProcessing Scramble {idx}...")
        run_single_dataset(iter_args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', required=True, help='Input CSV file (combined data)')
    ap.add_argument('--outdir', required=True, help='Output directory for plots')
    ap.add_argument('--num', default='ZllLumi', help='Numerator column')
    ap.add_argument('--numerr', default='ZllLumiErr', help='Numerator error column')
    ap.add_argument('--den', default='OffLumi', help='Denominator column')
    ap.add_argument('--t0', type=float, default=953551209.0, help='Reference time for phase')
    ap.add_argument('--nbins', type=int, default=100, help='Number of phase bins')
    ap.add_argument('--name', default='', help='Optional name prefix for output files')
    ap.add_argument('--n_scrambles', type=int, default=0, help='Process batch scrambles instead of single input')
    
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    if args.n_scrambles > 0:
        run_scrambles_batch(args)
    else:
        run_single_dataset(args)


if __name__ == '__main__':
    main()
