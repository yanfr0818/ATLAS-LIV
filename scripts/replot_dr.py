#!/usr/bin/env python
"""
replot_dr.py - Generate DR plots for scrambles with correct child seeds.

Reads child seeds from batch results JSON and generates DR plots.
Also computes unbinned mean/RMS for consistent statistics across scrambles.
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    results_path = Path("output/results/batch_1000_results.json")
    scrambles_dir = Path("output/scrambles_pq")
    plots_dir = Path("output/plots_DR")
    
    if not results_path.exists():
        print(f"Error: {results_path} not found!")
        return 1
    
    with open(results_path, "r") as f:
        data = json.load(f)
    
    master_seed = data["seed"]
    scrambles = data["scrambles"]
    
    print(f"Master seed: {master_seed}")
    print(f"Total scrambles: {len(scrambles)}")
    print()
    
    # Generate plots for first N scrambles
    n_plots = 5
    for i in range(n_plots):
        s = scrambles[i]
        idx = s["scramble_idx"]
        child_seed = s["child_seed"]
        
        infile = scrambles_dir / f"scramble_{idx:04d}.parquet"
        if not infile.exists():
            print(f"Warning: {infile} not found, skipping")
            continue
        
        print(f"Scramble {idx}: child_seed = {child_seed}")
        
        cmd = [
            sys.executable, "scripts/run_double_ratio.py",
            "--infile", str(infile),
            "--outdir", str(plots_dir),
            "--seed", str(master_seed),
            "--child_seed", str(child_seed),
            "--scramble_idx", str(idx),
            "--name", f"scramble_{idx:04d}"
        ]
        
        subprocess.run(cmd, check=True)
    
    print(f"\nDone! Generated DR plots for {n_plots} scrambles in {plots_dir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
