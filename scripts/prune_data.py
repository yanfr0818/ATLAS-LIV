#!/usr/bin/env python
"""
prune_data.py

Read original Set3 CSVs, keep only necessary columns, and save as a single compressed Parquet file.
This significantly speeds up loading and scrambling.

Columns kept:
- Fixed (Identity/Time): FillNum, RunNum, LBNum, LBStart, LBEnd, LBLive, LBFull
- Analysis (Data): ZllLumi, ZllLumiErr, OffLumi
"""

import os
from pathlib import Path
import pandas as pd

# Define columns to keep
KEEP_COLS = [
    # Identity & Time
    'FillNum', 'RunNum', 'LBNum', 'LBStart', 'LBEnd', 'LBLive', 'LBFull',
    # Data
    'ZllLumi', 'ZllLumiErr', 'OffLumi'
]

def main():
    # Source Directory (Hardcoded based on config)
    src_dir = Path(r"D:\HEP\ATLAS\LIV\Set3")
    years = ['2015', '2016', '2017', '2018']
    
    dfs = []
    print(f"Reading from {src_dir}...")
    
    for y in years:
        csv_path = src_dir / f"data{y}_shuffled_3.csv"
        if csv_path.exists():
            print(f"  Loading {csv_path.name} ...")
            # Read only necessary columns
            try:
                df = pd.read_csv(csv_path, usecols=lambda c: c in KEEP_COLS)
                # Ensure column order
                df = df[[c for c in KEEP_COLS if c in df.columns]]
                print(f"    -> {len(df)} rows, {len(df.columns)} cols")
                dfs.append(df)
            except Exception as e:
                print(f"    Error reading {csv_path}: {e}")
        else:
            print(f"  Warning: {csv_path} not found.")

    if not dfs:
        print("No data found!")
        return

    # Combine
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows")
    
    # Save to 'input' folder in scratch
    out_dir = Path("input")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "set3_pruned.parquet"
    
    print(f"Saving to {out_path} ...")
    combined.to_parquet(out_path, index=False, compression='zstd')
    
    # Verify
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done. File size: {size_mb:.2f} MB")

if __name__ == '__main__':
    main()
