#!/usr/bin/env python
"""
set3_scramble.py

Scramble Set3 Run II data (across-year shuffling):
- Combine all years into a single dataset
- Keep columns 1-7 fixed (time/LB identity)
- Shuffle remaining columns as a block across all rows (all years mixed)

Usage:
    python set3_scramble.py --indir <path_to_Set3> --outdir <output_path> --n 10 --seed 12345
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

YEARS = ["2015", "2016", "2017", "2018"]
FIXED_COLS = ["FillNum", "RunNum", "LBNum", "LBStart", "LBEnd", "LBLive", "LBFull"]


def load_and_combine_set3(indir: str) -> pd.DataFrame:
    """Load Set3 CSV files for all years and combine into single DataFrame."""
    dfs = []
    inpath = Path(indir)
    for y in YEARS:
        fname = f"data{y}_shuffled_3.csv"
        fpath = inpath / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            print(f"Loaded {fname}: {len(df)} rows")
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined total: {len(combined)} rows")
    return combined


def scramble_one(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Scramble: keep FIXED_COLS in place, shuffle all other columns as a block.
    This shuffles across ALL rows (all years mixed together).
    """
    all_cols = list(df.columns)
    payload_cols = [c for c in all_cols if c not in FIXED_COLS]
    
    perm = rng.permutation(len(df))
    result = df[FIXED_COLS].copy()
    payload = df[payload_cols].iloc[perm].reset_index(drop=True)
    
    return pd.concat([result, payload], axis=1)



def iter_scrambles(base: pd.DataFrame, n: int, seed: int, direct_seed: int = None):
    """Generate n scrambles with reproducible seeding."""
    if direct_seed is not None:
        # Direct mode: single scramble with explicit seed
        rng = np.random.default_rng(direct_seed)
        yield 0, scramble_one(base, rng)
        return

    ss_master = np.random.SeedSequence(seed)
    ss_samples = ss_master.spawn(n)
    
    for i, ss in enumerate(ss_samples):
        # Use entropy integer to ensure direct reproducibility (matches direct_seed mode)
        seed_int = int(ss.generate_state(1)[0])
        rng = np.random.default_rng(seed_int)
        yield i, scramble_one(base, rng)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory with Set3 CSV files")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--n", type=int, default=10, help="Number of scrambles")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed (master seed)")
    ap.add_argument("--direct_seed", type=int, default=None, 
                    help="Use this specific integer as seed (overrides --seed and --n to 1)")
    args = ap.parse_args()
    
    if args.direct_seed is not None:
        print(f"Using DIRECT SEED: {args.direct_seed} (generating 1 scramble)")
        args.n = 1
    
    base = load_and_combine_set3(args.indir)
    if base.empty:
        print("No data loaded, exiting.")
        return
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Write manifest
    manifest = {
        "n_scrambles": args.n,
        "seed": args.seed,
        "fixed_columns": FIXED_COLS,
        "method": "across_year",
        "total_rows": len(base),
        "source_years": YEARS,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    
    # Generate scrambles - each is a single combined CSV
    for i, scr_df in iter_scrambles(base, args.n, args.seed, args.direct_seed):
        out_path = outdir / f"scramble_{i:04d}.csv"
        scr_df.to_csv(out_path, index=False)
        print(f"Generated scramble {i+1}/{args.n}: {out_path.name}")
    
    print(f"Done. {args.n} scrambles in {outdir}")


if __name__ == "__main__":
    main()
