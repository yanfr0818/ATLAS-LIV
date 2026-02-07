#!/usr/bin/env python
"""
run_batch_analysis.py

Optimized batch analysis for 1000 scrambles:
- Uses Parquet format for efficient storage (optional)
- Column pruning to reduce data size
- In-memory processing option (no disk I/O for scrambles)
- Generates all SME fits and saves results

Usage:
    # In-memory (recommended - fastest, no disk space needed)
    python run_batch_analysis.py --n 1000 --seed 12345 --mode memory
    
    # With Parquet export (for later re-analysis)
    python run_batch_analysis.py --n 1000 --seed 12345 --mode parquet --export_dir output/scrambles_pq
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sme_fit import fit_all_sme_coeffs, fit_results_to_dict, SME_COEFFICIENTS

# Columns needed for analysis
ANALYSIS_COLS = ['LBStart', 'LBEnd', 'ZllLumi', 'ZllLumiErr', 'OffLumi']

# Fixed columns (time/LB identity) - not shuffled
FIXED_COLS = ['FillNum', 'RunNum', 'LBNum', 'LBStart', 'LBEnd', 'LBLive', 'LBFull']

# Sidereal day in hours
SIDEREAL_DAY_H = 23.9344696


def load_original_data(path: str) -> pd.DataFrame:
    """Load pruned Set3 Parquet data."""
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"Pruned data not found at {fpath}")
    
    print(f"Loading pruned data from {fpath}...")
    df = pd.read_parquet(fpath)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} cols")
    return df


def scramble_one(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Scramble: keep FIXED_COLS in place, shuffle other columns as a block."""
    all_cols = list(df.columns)
    payload_cols = [c for c in all_cols if c not in FIXED_COLS]
    
    perm = rng.permutation(len(df))
    result = df[FIXED_COLS].copy()
    payload = df[payload_cols].iloc[perm].reset_index(drop=True)
    
    return pd.concat([result, payload], axis=1)


def fold_phase(tmid: np.ndarray, per_h: float) -> np.ndarray:
    """Fold time into phase [0, 1)."""
    return np.mod(tmid / 3600.0, per_h) / per_h


def compute_double_ratio_and_fit(
    df: pd.DataFrame,
    bin_sec: float = 840.0,
) -> dict:
    """Compute double ratio and fit all SME coefficients."""
    
    # Filter valid rows
    df = df[df['OffLumi'] > 0].copy()
    
    per_h = SIDEREAL_DAY_H
    per_sec = per_h * 3600.0
    nbins = int(np.round(per_sec / bin_sec))
    
    # Phase calculation
    tmid = 0.5 * (df['LBStart'].to_numpy() + df['LBEnd'].to_numpy())
    phi = fold_phase(tmid, per_h)
    
    # Bin assignment
    idx = np.floor(phi * nbins).astype(int)
    idx = np.clip(idx, 0, nbins - 1)
    
    # Accumulate sums
    n = np.zeros(nbins, dtype=float)
    nerr2 = np.zeros(nbins, dtype=float)
    d = np.zeros(nbins, dtype=float)
    
    nv = df['ZllLumi'].to_numpy(dtype=float)
    ev = df['ZllLumiErr'].to_numpy(dtype=float)
    dv = df['OffLumi'].to_numpy(dtype=float)
    
    np.add.at(n, idx, nv)
    np.add.at(nerr2, idx, ev * ev)
    np.add.at(d, idx, dv)
    
    # Calculate ratio
    ok = d > 0
    r = np.zeros(nbins)
    r[ok] = n[ok] / d[ok]
    r0 = n.sum() / d.sum()
    
    rd = np.full(nbins, np.nan)
    rd_err = np.full(nbins, np.nan)
    rd[ok] = r[ok] / r0
    rd_err[ok] = np.sqrt(nerr2[ok]) / (d[ok] * r0)
    
    phi_centers = (np.arange(nbins) + 0.5) / nbins
    
    # Fit SME coefficients
    fits = fit_all_sme_coeffs(phi_centers, rd, rd_err)
    
    return {
        'nbins': nbins,
        'nbins_filled': int(ok.sum()),
        'n_rows': len(df),
        'fits': fit_results_to_dict(fits),
    }


def run_batch_memory(
    original_df: pd.DataFrame,
    n: int,
    seed: int,
    bin_sec: float = 840.0,
) -> list:
    """Run batch analysis entirely in memory (most efficient)."""
    
    # Create explicit integer seeds for full reproducibility
    # Using SeedSequence to manage stream, but use entropy integer for init
    ss_master = np.random.SeedSequence(seed)
    # Spawn child sequences
    child_sequences = ss_master.spawn(n)
    
    results = []
    start_time = time.time()
    
    for i, ss_child in enumerate(child_sequences):
        # generate_state(1) returns array of uint64, take first one
        child_seed = int(ss_child.generate_state(1)[0])
        # Use the explicit integer seed
        rng = np.random.default_rng(child_seed)
        scrambled = scramble_one(original_df, rng)
        
        result = compute_double_ratio_and_fit(scrambled, bin_sec)
        result['scramble_idx'] = i
        result['master_seed'] = seed
        result['child_seed'] = int(child_seed)  # Save specific seed
        results.append(result)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"Processed {i+1}/{n} ({rate:.1f}/s, ETA: {eta:.0f}s)")
    
    return results


def run_batch_parquet(
    original_df: pd.DataFrame,
    n: int,
    seed: int,
    export_dir: str,
    bin_sec: float = 840.0,
) -> list:
    """Run batch analysis and export scrambles to Parquet."""
    
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Use only analysis columns for export (much smaller)
    export_cols = FIXED_COLS + [c for c in ANALYSIS_COLS if c not in FIXED_COLS]
    
    # Create explicit integer seeds using SeedSequence
    ss_master = np.random.SeedSequence(seed)
    child_sequences = ss_master.spawn(n)
    
    results = []
    start_time = time.time()
    
    for i, ss_child in enumerate(child_sequences):
        child_seed = int(ss_child.generate_state(1)[0])
        rng = np.random.default_rng(child_seed)
        scrambled = scramble_one(original_df, rng)
        
        # Export to Parquet (pruned columns)
        scrambled[export_cols].to_parquet(
            export_path / f"scramble_{i:04d}.parquet",
            index=False,
            compression='zstd'
        )
        
        # Analyze
        result = compute_double_ratio_and_fit(scrambled, bin_sec)
        result['scramble_idx'] = i
        result['master_seed'] = seed
        result['child_seed'] = int(child_seed)
        results.append(result)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"Processed {i+1}/{n} ({rate:.1f}/s, ETA: {eta:.0f}s)")
    
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_file', default='input/set3_pruned.parquet',
                    help='Path to pruned Set3 Parquet file')
    ap.add_argument('--n', type=int, default=1000, help='Number of scrambles')
    ap.add_argument('--seed', type=int, default=12345, help='Random seed')
    ap.add_argument('--mode', choices=['memory', 'parquet'], default='memory',
                    help='Processing mode: memory (fastest) or parquet (exports files)')
    ap.add_argument('--export_dir', default='output/scrambles_pq',
                    help='Directory for Parquet export (if mode=parquet)')
    ap.add_argument('--outdir', default='output/results',
                    help='Output directory for results')
    ap.add_argument('--bin_sec', type=float, default=840.0,
                    help='Phase bin size in seconds')
    args = ap.parse_args()
    
    # Load data
    # Load data
    original = load_original_data(args.input_file)
    
    # Run batch
    print(f"\nStarting {args.n} scrambles (mode={args.mode})...")
    
    if args.mode == 'memory':
        results = run_batch_memory(original, args.n, args.seed, args.bin_sec)
    else:
        results = run_batch_parquet(original, args.n, args.seed, args.export_dir, args.bin_sec)
    
    # Save results
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'seed': args.seed,
        'n_scrambles': args.n,
        'bin_sec': args.bin_sec,
        'mode': args.mode,
        'scrambles': results,
    }
    
    out_path = outdir / f"batch_{args.n}_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")
    
    # Quick summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for coeff in SME_COEFFICIENTS[:4]:  # Show first 4 as sample
        values = [r['fits'][coeff]['value'] for r in results]
        errors = [r['fits'][coeff]['error'] for r in results]
        sigs = [abs(v/e) for v, e in zip(values, errors)]
        
        print(f"{coeff}: mean={np.mean(values):.2e}, std={np.std(values):.2e}, "
              f"max_sig={max(sigs):.2f}σ")
    print("...")


if __name__ == '__main__':
    main()
