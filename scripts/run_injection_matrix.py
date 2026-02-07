#!/usr/bin/env python
"""
run_injection_matrix.py

Performs a comprehensive signal injection test:
1. For each of the 12 SME coefficients:
   - Inject a fixed large signal (e.g. 5 sigma or 1e-4) into a null scramble.
   - Run the full fit for all 12 coefficients.
   - Record the fitted values.
2. Generate a "Mixing Matrix" plot:
   - Rows = Injected Coefficient
   - Cols = Fitted Coefficient
   - Color = Fitted Value / Injected Value (Recovery Fraction)
   
This validates:
- Sensitivity (Diagonal elements should be ~1.0)
- Correlations (Off-diagonal elements show physical mixing)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add scripts dir to path
sys.path.append(str(Path(__file__).parent))

from sme_fit import SME_COEFFICIENTS
from inject_signal import inject_signal
from run_batch_analysis import compute_double_ratio_and_fit

def main():
    infile = "output/scrambles_pq/scramble_0000.parquet"
    outdir = Path("output/plots_analysis")
    outdir.mkdir(exist_ok=True)
    
    # Baseline: Compute null fit for N scrambles
    N_SCRAMBLES = 100
    INJECT_VAL = 1e-4
    print(f"Running injection matrix scan (val={INJECT_VAL:.1e}) averaged over {N_SCRAMBLES} scrambles...")
    
    # Store aggregated recoveries: [inj_idx, fit_idx, scramble_idx]
    recoveries = np.zeros((12, 12, N_SCRAMBLES))
    
    for s_idx in range(N_SCRAMBLES):
        infile = f"output/scrambles_pq/scramble_{s_idx:04d}.parquet"
        df = pd.read_parquet(infile)
        
        # Null fit for this scramble
        base_res = compute_double_ratio_and_fit(df)
        base_vals = {k: v['value'] for k, v in base_res['fits'].items()}
        
        for i, inj_name in enumerate(SME_COEFFICIENTS):
            # Inject
            df_inj = inject_signal(df, inj_name, INJECT_VAL)
            
            # Fit
            res = compute_double_ratio_and_fit(df_inj)
            
            for j, fit_name in enumerate(SME_COEFFICIENTS):
                fit_val = res['fits'][fit_name]['value']
                null_val = base_vals[fit_name]
                
                # Recovery
                rec = (fit_val - null_val) / INJECT_VAL
                recoveries[i, j, s_idx] = rec
        
        print(f"  Processed scramble {s_idx+1}/{N_SCRAMBLES}")

    # Average over scrambles
    matrix = np.mean(recoveries, axis=2)
    std_matrix = np.std(recoveries, axis=2)

    # Plotting
    print("Generating matrix plot...")
    plt.figure(figsize=(12, 10))
    
    # Mask diagonals < 0.1? No, show everything.
    
    im = plt.imshow(matrix, cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    plt.colorbar(im, label="Fitted / Injected")
    
    # Annotate
    for I in range(12):
        for J in range(12):
            val = matrix[I, J]
            color = "white" if abs(val) > 0.6 else "black"
            plt.text(J, I, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
            
    plt.xticks(np.arange(12), SME_COEFFICIENTS, rotation=45)
    plt.yticks(np.arange(12), SME_COEFFICIENTS)
    
    plt.title(f"SME Signal Injection Mixing Matrix\nNormalization: (Fitted - Null) / Injected", fontsize=14)
    plt.xlabel("Fitted Coefficient")
    plt.ylabel("Injected Coefficient")
    
    outpath = outdir / "signal_injection_matrix.pdf"
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Wrote {outpath}")

if __name__ == '__main__':
    main()
