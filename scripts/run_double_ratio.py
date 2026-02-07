#!/usr/bin/env python
"""
run_double_ratio.py

Compute double ratio from Set3 data and generate plots.
Simplified design:
- Reads single combined CSV (no year separation)
- Uses direct phase binning with configurable bin duration per period
- No rebinning - bins are calculated directly at desired resolution

Usage:
    python run_double_ratio.py --infile <data.csv> --outdir <plot_path>
    python run_double_ratio.py --infile <data.csv> --outdir <plot_path> --bin_sday 840 --bin_day 900 --bin_hour 60
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Period durations in hours
PER_H = {'sday': 23.9344696, 'day': 24.0, 'hour': 1.0}


def fold_phase(tmid: np.ndarray, per_h: float, t0: float = 0.0) -> np.ndarray:
    """Fold time into phase [0, 1) for a given period."""
    return np.mod((tmid - t0) / 3600.0, per_h) / per_h


def phase_binned_double_ratio(
    df: pd.DataFrame,
    num_col: str,
    numerr_col: str,
    den_col: str,
    per_tag: str,
    nbins: int = 100,
    t0: float = 0.0
) -> tuple:
    """
    Calculate phase-binned double ratio with direct binning.
    
    Args:
        df: DataFrame with LBStart, LBEnd, and the num/den columns
        num_col: Column name for numerator (e.g., 'ZllLumi')
        numerr_col: Column name for numerator error (e.g., 'ZllLumiErr')
        den_col: Column name for denominator (e.g., 'OffLumi')
        per_tag: Period tag ('sday', 'day', 'hour')
        nbins: Number of phase bins (default 100)
        t0: Reference time for phase calculation
        
    Returns:
        x: Phase bin centers [0, 1)
        y: Double ratio values (normalized to mean)
        yerr: Errors on double ratio
        info: Dict with binning metadata
    """
    per_h = PER_H[per_tag]
    per_sec = per_h * 3600.0
    bin_sec = per_sec / nbins
    
    # Calculate phase from LB midpoints
    tmid = 0.5 * (df['LBStart'].to_numpy() + df['LBEnd'].to_numpy())
    phi = fold_phase(tmid, per_h, t0)
    
    # Assign to bins
    idx = np.floor(phi * nbins).astype(int)
    idx = np.clip(idx, 0, nbins - 1)  # Handle edge case phi=1.0
    
    # Accumulate sums per bin
    n = np.zeros(nbins, dtype=float)
    nerr2 = np.zeros(nbins, dtype=float)
    d = np.zeros(nbins, dtype=float)
    
    nv = df[num_col].to_numpy(dtype=float)
    ev = df[numerr_col].to_numpy(dtype=float)
    dv = df[den_col].to_numpy(dtype=float)
    
    np.add.at(n, idx, nv)
    np.add.at(nerr2, idx, ev * ev)
    np.add.at(d, idx, dv)
    
    # Filter empty bins
    ok = d > 0
    n = n[ok]
    nerr2 = nerr2[ok]
    d = d[ok]
    
    # Calculate ratio and normalize
    r = n / d
    r0 = n.sum() / d.sum()
    y = r / r0 - 1.0
    yerr = np.sqrt(nerr2) / (d * r0)
    
    # Bin centers as phase
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


def make_plot(outpath: str, title: str, top_text: str, bot_text: str, series: list, y_pref=(0.99, 1.01)):
    """Generate double ratio plot with multiple series."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    
    handles = []
    labels = []
    for s in series:
        h = ax.errorbar(s['x'], s['y'], yerr=s['yerr'], 
                       fmt=s['fmt'], markersize=3, elinewidth=1, capsize=0, linewidth=0)
        handles.append(h[0])
        labels.append(s['label'])
    
    ax.axhline(0.0, linewidth=1.0, color='black')
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('phase')
    ax.set_ylabel('Double Ratio - 1')
    ax.set_title(title)
    
    # Dynamic y-limits
    ylo = min(float(np.min(s['y'] - s['yerr'])) for s in series)
    yhi = max(float(np.max(s['y'] + s['yerr'])) for s in series)
    lo, hi = y_pref
    if ylo >= lo and yhi <= hi:
        ax.set_ylim(lo, hi)
    else:
        pad = 0.08 * (yhi - ylo) if yhi > ylo else 0.002
        ax.set_ylim(ylo - pad, yhi + pad)
    
    ax.text(0.02, 0.98, top_text, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, linewidth=0.5))
    ax.text(0.02, 0.06, bot_text, transform=ax.transAxes, va='bottom', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, linewidth=0.5))
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def make_single_period_plot(outpath: str, per_tag: str, x, y, yerr, info: dict, 
                             top_text: str, y_pref=(-0.005, 0.005)):
    """Generate single-period double ratio plot with mean/RMS in legend."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Period-specific colors (same as combined plot)
    color_map = {'sday': 'C0', 'day': 'C1', 'hour': 'C2'}  # default matplotlib colors
    fmt_map = {'sday': 'o', 'day': 's', 'hour': 'D'}
    label_map = {'sday': 'Sidereal Day', 'day': 'Solar Day', 'hour': '1 Hour'}
    
    color = color_map.get(per_tag, 'black')
    fmt = fmt_map.get(per_tag, 'o')
    per_label = label_map.get(per_tag, per_tag)
    
    # Calculate statistics
    mean_val = np.mean(y)
    rms_val = np.std(y)
    
    # Data points with period-specific color
    ax.errorbar(x, y, yerr=yerr, fmt=fmt, color=color, markersize=4, 
                elinewidth=1, capsize=2, label=f'{per_label}')
    
    # Ideal line (y=0)
    ax.axhline(0.0, linewidth=1.5, color='red', linestyle='-', label='Ideal (DR-1 = 0)')
    
    # Mean line
    ax.axhline(mean_val, linewidth=1.2, color='blue', linestyle='--', 
               label=f'Mean = {mean_val:.2e}')
    
    # RMS bands (dashed lines at ±RMS from mean)
    ax.axhline(mean_val + rms_val, linewidth=1.0, color='green', linestyle=':', 
               label=f'±RMS = {rms_val:.2e}')
    ax.axhline(mean_val - rms_val, linewidth=1.0, color='green', linestyle=':')
    
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('Phase', fontsize=12)
    ax.set_ylabel('Double Ratio - 1', fontsize=12)
    
    # Period-specific title
    period_titles = {'sday': 'Sidereal Day (23.93h)', 'day': 'Solar Day (24h)', 'hour': '1 Hour'}
    ax.set_title(f'Double Ratio vs Phase: {period_titles.get(per_tag, per_tag)}', fontsize=14)
    
    # Dynamic y-limits
    ylo = float(np.min(y - yerr))
    yhi = float(np.max(y + yerr))
    lo, hi = y_pref
    if ylo >= lo and yhi <= hi:
        ax.set_ylim(lo, hi)
    else:
        pad = 0.1 * (yhi - ylo) if yhi > ylo else 0.002
        ax.set_ylim(ylo - pad, yhi + pad)
    
    # Info text box
    info_text = f"Bins: {info['nbins']} @ {info['bin_min']:.1f} min ({info['nbins_filled']} filled)"
    ax.text(0.02, 0.98, top_text, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, linewidth=0.5))
    ax.text(0.02, 0.06, info_text, transform=ax.transAxes, va='bottom', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, linewidth=0.5))
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', required=True, help='Input CSV file (combined data)')
    ap.add_argument('--outdir', required=True, help='Output directory for plots')
    ap.add_argument('--num', default='ZllLumi', help='Numerator column')
    ap.add_argument('--numerr', default='ZllLumiErr', help='Numerator error column')
    ap.add_argument('--den', default='OffLumi', help='Denominator column')
    ap.add_argument('--t0', type=float, default=0.0, help='Reference time for phase')
    
    # Number of phase bins (fixed for all periods)
    ap.add_argument('--nbins', type=int, default=100, help='Number of phase bins')
    
    # Output options
    ap.add_argument('--name', default='', help='Optional name prefix for output files')
    
    # Seed display (optional - for scrambled data only)
    ap.add_argument('--seed', type=int, default=None, help='Master seed used for scrambling (for display)')
    ap.add_argument('--scramble_idx', type=int, default=None, help='Scramble index (for display)')
    ap.add_argument('--child_seed', type=int, default=None, help='Explicit child seed (for display)')
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data
    usecols = ['LBStart', 'LBEnd', args.den, args.num, args.numerr]
    if args.infile.endswith('.parquet'):
        df = pd.read_parquet(args.infile, columns=usecols)
    else:
        df = pd.read_csv(args.infile, usecols=usecols)
    df = df[df[args.den] > 0]
    print(f"Loaded {len(df)} rows from {args.infile}")
    
    # Period configuration (all use same nbins)
    periods = [
        ('sday', 'o', 'sday'),
        ('day', 's', 'day'),
        ('hour', 'D', '1h'),
    ]
    
    # Calculate double ratio for each period
    series = []
    bot_lines = []
    period_data = []  # Store data for individual plots
    for per_tag, fmt, label in periods:
        x, y, yerr, info = phase_binned_double_ratio(
            df, args.num, args.numerr, args.den, per_tag, args.nbins, args.t0
        )
        series.append({'x': x, 'y': y, 'yerr': yerr, 'fmt': fmt, 'label': label})
        bot_lines.append(f"{per_tag}: {info['nbins']} bins @ {info['bin_min']:.1f} min ({info['nbins_filled']} filled)")
        period_data.append((per_tag, label, x, y, yerr, info))
    
    # Generate plot
    # Row 1: Metadata
    top_parts = [f"num={args.num}, den={args.den}, rows={len(df)}"]
    top = ", ".join(top_parts)
    
    # Row 2: Seed info
    seed_text = ""
    if args.seed is not None:
        seed_parts = [f"master={args.seed}"]
        if args.child_seed is not None:
            seed_parts.append(f"child={args.child_seed}")
        if args.scramble_idx is not None:
            seed_parts.append(f"idx={args.scramble_idx}")
        seed_text = ", ".join(seed_parts)
    elif args.child_seed is not None:
         seed_text = f"child={args.child_seed}"
         
    bot = "\n".join(bot_lines)

    # Pass metadata as top_text, seed info as a second line (concatenated with newline)
    if seed_text:
        top_combo = f"{top}\n{seed_text}"
    else:
        top_combo = top

    name_prefix = f"{args.name}_" if args.name else ""
    
    # Generate SEPARATE plots for each period
    for per_tag, label, x, y, yerr, info in period_data:
        outpath_single = os.path.join(args.outdir, f"{name_prefix}double_ratio_{per_tag}.pdf")
        make_single_period_plot(outpath_single, per_tag, x, y, yerr, info, top_combo)
    
    # Also generate combined plot
    outpath = os.path.join(args.outdir, f"{name_prefix}double_ratio.pdf")
    
    # Set standard limits for scatter plots around 0
    make_plot(outpath, "Double Ratio - 1 vs Phase", top_combo, bot, series, y_pref=(-0.005, 0.005))
    print(f"Wrote {outpath}")


if __name__ == '__main__':
    main()
