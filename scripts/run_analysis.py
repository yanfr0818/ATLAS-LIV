#!/usr/bin/env python
"""
run_analysis.py

Complete LIV analysis pipeline:
1. Read CSV data (combined from all years)
2. Compute phase-binned double ratio for sidereal period
3. Fit all SME coefficients
4. Output results to JSON

Usage:
    # Single file analysis
    python run_analysis.py --infile data.csv --outdir output/results
    
    # Batch analysis of all scrambles
    python run_analysis.py --batch --scramble_dir output/scrambles --outdir output/results
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Import our modules
from sme_fit import fit_all_sme_coeffs, fit_results_to_dict, print_fit_summary

# Period duration for sidereal day
SIDEREAL_DAY_HOURS = 23.9344696


def fold_phase(tmid: np.ndarray, per_h: float, t0: float = 0.0) -> np.ndarray:
    """Fold time into phase [0, 1) for a given period."""
    return np.mod((tmid - t0) / 3600.0, per_h) / per_h


def compute_double_ratio(
    df: pd.DataFrame,
    num_col: str = 'ZllLumi',
    numerr_col: str = 'ZllLumiErr',
    den_col: str = 'OffLumi',
    bin_sec: float = 840.0,
    t0: float = 0.0,
) -> dict:
    """
    Compute phase-binned double ratio for sidereal period.
    
    Returns dict with phi, rd, rd_err arrays plus metadata.
    """
    per_h = SIDEREAL_DAY_HOURS
    per_sec = per_h * 3600.0
    nbins = int(np.round(per_sec / bin_sec))
    
    # Calculate phase from LB midpoints
    tmid = 0.5 * (df['LBStart'].to_numpy() + df['LBEnd'].to_numpy())
    phi = fold_phase(tmid, per_h, t0)
    
    # Assign to bins
    idx = np.floor(phi * nbins).astype(int)
    idx = np.clip(idx, 0, nbins - 1)
    
    # Accumulate sums
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
    
    # Calculate ratio and normalize
    r = np.zeros(nbins)
    r[ok] = n[ok] / d[ok]
    r0 = n.sum() / d.sum()
    
    rd = np.full(nbins, np.nan)
    rd_err = np.full(nbins, np.nan)
    
    rd[ok] = r[ok] / r0
    rd_err[ok] = np.sqrt(nerr2[ok]) / (d[ok] * r0)
    
    phi_centers = (np.arange(nbins) + 0.5) / nbins
    
    return {
        'phi': phi_centers,
        'rd': rd,
        'rd_err': rd_err,
        'nbins': nbins,
        'nbins_filled': int(ok.sum()),
        'bin_sec': bin_sec,
        'n_rows': len(df),
    }


def analyze_single(
    infile: str,
    outdir: str = None,
    name: str = None,
    seed: int = None,
    scramble_idx: int = None,
    bin_sec: float = 840.0,
    verbose: bool = True,
) -> dict:
    """
    Run full analysis on a single CSV file.
    
    Returns dict with double ratio data and SME fit results.
    """
    # Load data
    usecols = ['LBStart', 'LBEnd', 'OffLumi', 'ZllLumi', 'ZllLumiErr']
    df = pd.read_csv(infile, usecols=usecols)
    df = df[df['OffLumi'] > 0]
    
    if verbose:
        print(f"Loaded {len(df)} rows from {infile}")
    
    # Compute double ratio
    dr = compute_double_ratio(df, bin_sec=bin_sec)
    
    # Fit SME coefficients
    sme_results = fit_all_sme_coeffs(dr['phi'], dr['rd'], dr['rd_err'])
    
    if verbose:
        print_fit_summary(sme_results)
    
    # Build result structure
    result = {
        'input_file': str(infile),
        'name': name,
        'seed': seed,
        'scramble_idx': scramble_idx,
        'n_rows': dr['n_rows'],
        'nbins': dr['nbins'],
        'nbins_filled': dr['nbins_filled'],
        'bin_sec': dr['bin_sec'],
        'sme_fits': fit_results_to_dict(sme_results),
    }
    
    # Save if output dir specified
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_name = name if name else Path(infile).stem
        out_path = os.path.join(outdir, f"{out_name}_results.json")
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"Wrote {out_path}")
    
    return result


def analyze_batch(
    scramble_dir: str,
    outdir: str,
    manifest_path: str = None,
    bin_sec: float = 840.0,
) -> list:
    """
    Run analysis on all scrambles in a directory.
    
    Returns list of result dicts.
    """
    scramble_dir = Path(scramble_dir)
    
    # Load manifest if exists
    manifest_file = manifest_path or (scramble_dir / "manifest.json")
    if Path(manifest_file).exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
        seed = manifest.get('seed')
        n_scrambles = manifest.get('n_scrambles')
        print(f"Found manifest: seed={seed}, n_scrambles={n_scrambles}")
    else:
        seed = None
        n_scrambles = None
    
    # Find all scramble CSV files
    csv_files = sorted(scramble_dir.glob("scramble_*.csv"))
    print(f"Found {len(csv_files)} scramble files")
    
    all_results = []
    for csv_file in csv_files:
        # Extract scramble index from filename
        name = csv_file.stem
        try:
            idx = int(name.split('_')[1])
        except (IndexError, ValueError):
            idx = None
        
        result = analyze_single(
            infile=str(csv_file),
            outdir=outdir,
            name=name,
            seed=seed,
            scramble_idx=idx,
            bin_sec=bin_sec,
            verbose=True,
        )
        all_results.append(result)
    
    # Save summary
    summary_path = os.path.join(outdir, "batch_summary.json")
    summary = {
        'seed': seed,
        'n_scrambles': len(all_results),
        'bin_sec': bin_sec,
        'scrambles': [
            {
                'name': r['name'],
                'scramble_idx': r['scramble_idx'],
                'sme_fits': r['sme_fits'],
            }
            for r in all_results
        ],
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote batch summary to {summary_path}")
    
    return all_results


def main():
    ap = argparse.ArgumentParser(description="LIV analysis: double ratio + SME fitting")
    
    # Mode selection
    ap.add_argument('--batch', action='store_true', help='Batch mode for multiple scrambles')
    
    # Single file mode
    ap.add_argument('--infile', help='Input CSV file (single mode)')
    ap.add_argument('--name', default='', help='Name for output files')
    ap.add_argument('--seed', type=int, default=None, help='Seed for display')
    ap.add_argument('--scramble_idx', type=int, default=None, help='Scramble index')
    
    # Batch mode
    ap.add_argument('--scramble_dir', help='Directory with scramble CSV files')
    
    # Common options
    ap.add_argument('--outdir', default='output/results', help='Output directory')
    ap.add_argument('--bin_sec', type=float, default=840.0, help='Phase bin size in seconds')
    
    args = ap.parse_args()
    
    if args.batch:
        if not args.scramble_dir:
            print("Error: --scramble_dir required for batch mode")
            return
        analyze_batch(
            scramble_dir=args.scramble_dir,
            outdir=args.outdir,
            bin_sec=args.bin_sec,
        )
    else:
        if not args.infile:
            print("Error: --infile required for single mode")
            return
        analyze_single(
            infile=args.infile,
            outdir=args.outdir,
            name=args.name,
            seed=args.seed,
            scramble_idx=args.scramble_idx,
            bin_sec=args.bin_sec,
        )


if __name__ == '__main__':
    main()
