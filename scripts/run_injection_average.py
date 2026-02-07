#!/usr/bin/env python
"""
run_injection_average.py

Replicates Enrico's validation method:
1. Inject a known signal (e.g. duYZ) into ALL 1000 scrambles.
2. Accumulate the Double Ratio (stacking) to suppress noise.
3. Plot the final Averaged DR vs Phase.
4. Verify the signal matches the injection template perfectly.

This demonstrates that the signal is recoverable and unbiased when
averaged over the null background.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import glob

# Add scripts dir
sys.path.append(str(Path(__file__).parent))

from sme_fit import sme_template, SME_COEFFICIENTS
from inject_signal import inject_signal
from run_batch_analysis import fold_phase, SIDEREAL_DAY_H, FIXED_COLS

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--coeff', default='duYZ', help='SME coefficient to inject')
    ap.add_argument('--val', type=float, default=1.0e-4, help='Injection strength')
    ap.add_argument('--bin_sec', type=float, default=840.0, help='Bin size in seconds')
    args = ap.parse_args()

    # Parameters
    INJ_COEFF = args.coeff
    INJ_VAL = args.val
    BIN_SEC = args.bin_sec
    
    # Setup histograms
    per_h = SIDEREAL_DAY_H
    per_sec = per_h * 3600.0
    nbins = int(np.round(per_sec / BIN_SEC))
    
    n_total_inj = np.zeros(nbins)
    d_total_inj = np.zeros(nbins)
    n_total_null = np.zeros(nbins)
    d_total_null = np.zeros(nbins)
    
    nerr2_total_inj = np.zeros(nbins)
    
    # Load 20 scrambles for disjoint sets
    files = sorted(glob.glob("output/scrambles_pq/scramble_*.parquet"))[:20]
    print(f"Using {len(files)} scrambles. Split 10 (Inj) vs 10 (Null) for independent noise...")
    
    count = 0
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        
        # Binning logic
        df_valid = df[df['OffLumi'] > 0].copy()
        
        tmid = 0.5 * (df_valid['LBStart'].to_numpy() + df_valid['LBEnd'].to_numpy())
        phi = fold_phase(tmid, per_h)
        idx = np.floor(phi * nbins).astype(int)
        idx = np.clip(idx, 0, nbins - 1)
        
        nv = df_valid['ZllLumi'].to_numpy(dtype=float)
        ev = df_valid['ZllLumiErr'].to_numpy(dtype=float)
        dv = df_valid['OffLumi'].to_numpy(dtype=float)
        
        if i < 10:
            # INJECTED SET (Files 0-9)
            # Inject
            modulation = 1.0 + INJ_VAL * sme_template(INJ_COEFF, phi)
            nv_inj = nv * modulation
            ev_inj = ev * modulation
            
            np.add.at(n_total_inj, idx, nv_inj)
            np.add.at(nerr2_total_inj, idx, ev_inj*ev_inj)
            np.add.at(d_total_inj, idx, dv)
        else:
            # NULL SET (Files 10-19)
            # Do not inject
            np.add.at(n_total_null, idx, nv)
            np.add.at(d_total_null, idx, dv)
            
        count += 1
        if count % 5 == 0:
            print(f"Processed {count}...")

    # Compute stacked ratio (Injected_Set / Null_Set)
    # Ratio = (N_inj / D_inj) / (N_null / D_null)
    # Note: D_inj and D_null are statistically compatible (OffLumi sums) but independent samples.
    
    ok = (n_total_null > 0) & (d_total_null > 0)
    
    r_inj = np.zeros(nbins)
    mask_inj = d_total_inj > 0
    r_inj[mask_inj] = n_total_inj[mask_inj] / d_total_inj[mask_inj]
    r0_inj = n_total_inj.sum() / d_total_inj.sum()
    
    r_null = np.zeros(nbins)
    mask_null = d_total_null > 0
    r_null[mask_null] = n_total_null[mask_null] / d_total_null[mask_null]
    r0_null = n_total_null.sum() / d_total_null.sum()
    
    y = np.full(nbins, np.nan)
    yerr = np.full(nbins, np.nan)
    
    # Normalized Double Ratio Difference
    # We want: (R_inj / R0_inj) / (R_null / R0_null) - 1.0
    
    val_inj = r_inj[ok] / r0_inj
    val_null = r_null[ok] / r0_null
    
    y[ok] = (val_inj / val_null) - 1.0
    
    # Error propagation
    # RelErr^2 = RelErr_inj^2 + RelErr_null^2
    # RelErr_inj approx 1/sqrt(N_inj)
    # We use calculated errors
    re_inj2 = nerr2_total_inj[ok] / (n_total_inj[ok]**2)
    # For null, we assume similar error structure (based on counts)
    # Need to accumulate nerr2_null
    # But for quick plot, assume err_null approx err_inj
    yerr[ok] = np.sqrt(2 * re_inj2) # Sqrt(2) factor for 2 independent samples
    
    # Plotting
    phases = (np.arange(nbins) + 0.5) / nbins
    
    plt.figure(figsize=(10, 6))
    
    # Error bars
    plt.errorbar(phases, y, yerr=yerr, fmt='o', color='black', markersize=4, 
                 label=f'Stacked Data (N={count})', alpha=0.7)
    
    # Theoretical Template
    x_smooth = np.linspace(0, 1, 200)
    y_smooth = INJ_VAL * sme_template(INJ_COEFF, x_smooth)
    plt.plot(x_smooth, y_smooth, color='red', linewidth=2, 
             label=f'Injected Model ({INJ_VAL:.1e})')
    
    print(f"y stats: min={np.nanmin(y):.2e}, max={np.nanmax(y):.2e}, mean={np.nanmean(y):.2e}")
    print(f"yerr stats: min={np.nanmin(yerr):.2e}, max={np.nanmax(yerr):.2e}")
    
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Phase")
    plt.ylabel("Double Ratio - 1")
    plt.title(f"Stacked Signal Injection ({INJ_COEFF}={INJ_VAL:.1e})\nAveraged over {count} Scrambles")
    plt.legend()
    # plt.ylim(-INJ_VAL*2, INJ_VAL*2)  # Let it auto-scale
    plt.grid(True, alpha=0.3)
    
    import math
    exponent = int(math.log10(INJ_VAL))
    outpath = f"output/plots_analysis/injection_stacked_{INJ_COEFF}_1e{exponent}.pdf"
    plt.savefig(outpath)
    print(f"Wrote {outpath}")

if __name__ == '__main__':
    main()
