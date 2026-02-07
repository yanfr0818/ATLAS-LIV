#!/usr/bin/env python
"""
combine_original.py

Combine original Set3 data from all years into a single CSV (no scrambling).
This creates the baseline for comparison with scrambled data.

Usage:
    python combine_original.py --indir <path_to_Set3> --outfile <output.csv>
"""

from pathlib import Path
import pandas as pd

YEARS = ["2015", "2016", "2017", "2018"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory with Set3 CSV files")
    ap.add_argument("--outfile", required=True, help="Output CSV file")
    args = ap.parse_args()
    
    dfs = []
    inpath = Path(args.indir)
    for y in YEARS:
        fname = f"data{y}_shuffled_3.csv"
        fpath = inpath / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            print(f"Loaded {fname}: {len(df)} rows")
            dfs.append(df)
    
    if not dfs:
        print("No data loaded, exiting.")
        return
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined total: {len(combined)} rows")
    
    outpath = Path(args.outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outpath, index=False)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
