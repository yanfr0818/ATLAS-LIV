#!/usr/bin/env python
"""
validate_neff.py

Validates the assumption that Effective Counts (N_eff = (Lumi/LumiErr)**2)
approximates the true Raw Counts (ZeeRaw + ZmumuRaw).

Methodology:
1. Load original CSV data from Set3/ (2015-2018).
2. Calculate N_raw = ZeeRaw + ZmumuRaw.
3. Calculate N_eff = (ZllLumi / ZllLumiErr)**2.
4. Calculate correlation and mean ratio.
5. Plot scatter N_raw vs N_eff.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def main():
    files = sorted(glob.glob("Set3/data*.csv"))
    print(f"Found {len(files)} files: {files}")
    
    dfs = []
    for f in files:
        print(f"Loading {f}...")
        df = pd.read_csv(f)
        # Filter negative or zero lumis/errors
        df = df[(df['ZllLumi'] > 0) & (df['ZllLumiErr'] > 0) & (df['ZeeRaw'] >= 0) & (df['ZmumuRaw'] >= 0)]
        dfs.append(df)
        
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df_all)}")
    
    # 1. Total Raw Counts
    n_raw = df_all['ZeeRaw'] + df_all['ZmumuRaw']
    
    # 2. Effective Counts
    # N_eff = (Val/Err)^2
    n_eff = (df_all['ZllLumi'] / df_all['ZllLumiErr'])**2
    
    # 3. Comparison
    ratio = n_eff / n_raw
    # Filter inf/nan
    valid = np.isfinite(ratio) & (n_raw > 0)
    
    n_raw = n_raw[valid]
    n_eff = n_eff[valid]
    ratio = ratio[valid]
    
    corr = np.corrcoef(n_raw, n_eff)[0,1]
    mean_ratio = np.mean(ratio)
    std_ratio = np.std(ratio)
    
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"Correlation (N_raw vs N_eff): {corr:.4f}")
    print(f"Ratio (N_eff / N_raw):        {mean_ratio:.4f} ± {std_ratio:.4f}")
    print(f"Approximation Quality:        {abs(mean_ratio - 1.0) * 100:.2f}% deviation")
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(n_raw, n_eff, alpha=0.05, s=1, label='Data LBs')
    
    # Ideal line
    max_val = max(n_raw.max(), n_eff.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Ideal (y=x)')
    
    ax.set_xlabel('True Raw Counts (ZeeRaw + ZmumuRaw)')
    ax.set_ylabel('Effective Counts ((L/Err)^2)')
    ax.set_title(f"Validation of Effective Counts\nRatio = {mean_ratio:.3f}, Corr = {corr:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    outpath = "validation_neff.pdf"
    plt.savefig(outpath)
    print(f"\nSaved plot to {outpath}")
    
    # Interpretation
    if abs(mean_ratio - 1.0) < 0.05: # 5% tolerance
        print("CONCLUSION: VALID. N_eff allows accurate Poisson fluctuation.")
    else:
        print("CONCLUSION: INVALID. N_eff deviates significantly from N_raw.")

if __name__ == "__main__":
    main()
