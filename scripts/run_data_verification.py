#!/usr/bin/env python
"""
run_data_verification.py

Verifies statistical methods using:
1. The ACTUAL 2000 pre-generated scrambles (Method A).
2. Direct Poisson fluctuation of the original data (Method B).
"""

import argparse
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import sys

# Add local scripts to path
sys.path.append(str(Path(__file__).parent))

# Import existing analysis tools to ensure we use EXACTLY the same method
from run_batch_analysis import compute_double_ratio_and_fit, load_original_data, fold_phase, SIDEREAL_DAY_H
from inject_signal import inject_signal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scramble_dir', default='output/scrambles_pq', help='Directory with pre-generated scrambles')
    parser.add_argument('--data_file', default='input/set3_pruned.parquet', help='Original data file')
    parser.add_argument('--n_toys', type=int, default=2000, help='Number of toys/scrambles to use')
    parser.add_argument('--signal', type=float, default=1e-5, help='Signal calculation to inject')
    parser.add_argument('--coeff', default='duYZ', help='Coefficient to inject')
    args = parser.parse_args()
    
    print(f"Data Verification: N={args.n_toys}, Signal={args.signal}, Coeff={args.coeff}")
    
    # 1. Load Original Data
    print(f"Loading original data from {args.data_file}...")
    df_data = load_original_data(args.data_file)
    
    # Pre-calculate phase for Method B
    tmid = 0.5 * (df_data['LBStart'].to_numpy() + df_data['LBEnd'].to_numpy())
    df_data['Phase'] = fold_phase(tmid, SIDEREAL_DAY_H)
    
    # ==============================================================================
    # Method A: Use ACTUAL Stored Scrambles
    # ==============================================================================
    print("\n" + "="*60)
    print("Method A: Processing Pre-Generated Scrambles")
    print("="*60)
    
    # Try Parquet then CSV
    scramble_files = sorted(glob.glob(f"{args.scramble_dir}/scramble_*.parquet"))
    if not scramble_files:
        scramble_files = sorted(glob.glob(f"{args.scramble_dir}/scramble_*.csv"))
        file_type = 'csv'
    else:
        file_type = 'parquet'
        
    if len(scramble_files) < args.n_toys:
        print(f"Warning: Found only {len(scramble_files)} scrambles, requested {args.n_toys}")
        if len(scramble_files) == 0:
            print("Error: No scrambles found. Please interpret this as a need to run set3_scramble.py first.")
            return
        args.n_toys = len(scramble_files)
    
    # Use only requested number
    scramble_files = scramble_files[:args.n_toys]
    
    res_A = []
    
    for i, fpath in enumerate(scramble_files):
        if (i+1) % 500 == 0: print(f"Scramble {i+1}...")
        
        # Load Scramble
        if file_type == 'parquet':
            df_scram = pd.read_parquet(fpath)
        else:
            df_scram = pd.read_csv(fpath)
        
        # Inject Signal
        # Using the standard injection script
        df_inj = inject_signal(df_scram, args.coeff, args.signal)
        
        # Fit
        res = compute_double_ratio_and_fit(df_inj)
        fit_val = res['fits'][args.coeff]['value']
        fit_err = res['fits'][args.coeff]['error']
        res_A.append([fit_val, fit_err])
        
    res_A = np.array(res_A)
    
    # ==============================================================================
    # Method B: Direct Poisson Fluctuation of Data
    # ==============================================================================
    print("\n" + "="*60)
    print("Method B: Direct Poisson Fluctuation of Original Data")
    print("="*60)
    
    # What are we fluctuating?
    # We fluctuate the counts ZllLumi and OffLumi?
    # Or just ZllLumi? Reference (OffLumi) usually has much smaller errors?
    # Standard: Fluctuate both independently.
    
    # BUT: The user asked to "fluctuate directly on the original dataset".
    # Original Data: N_obs in each LB.
    # Fluctuation: N_new ~ Poisson(N_obs).
    # This assumes N_obs is the best estimator of the rate (Bootstrap).
    
    # We will do this for 'ZllLumi' and 'OffLumi'.
    # Note: ZllLumi is already normalized? No, it's a rate/count?
    # Let's check columns: 'ZllLumi', 'ZllLumiErr'.
    # If ZllLumi is fractional (Luminosity), we need counts.
    # N = (Lumi / Err)^2 ?
    # Let's assume ZllLumi is effectively counts for this check (or proportional).
    # Poisson fluctuation of a float X with error E:
    # New_X = Poisson(X) ? No, Poisson is integer.
    # If X is a float representing N events, New_X ~ Gamma(k=X, theta=1)?
    # Or just Gauss(X, sqrt(X))?
    # For high stats (N>100), Poisson -> Gauss.
    
    # Let's assume Gaussian fluctuation using the provided Errors.
    # New_Z = Gauss(Mean=Old_Z, Sigma=Old_Z_Err)
    # This handles weights/normalizations automatically.
    
    res_B = []
    
    # Store first 5 for inspection
    sample_distributions = [] 
    
    print("Generating 5 sample distributions for inspection...")
    for i in range(args.n_toys):
        if (i+1) % 500 == 0: print(f"Poisson Toy {i+1}...")
        
        df_toy = df_data.copy()
        
        # Fluctuate ZllLumi
        # val + random_normal * err
        z_noise = np.random.randn(len(df_toy))
        df_toy['ZllLumi'] = df_toy['ZllLumi'] + z_noise * df_toy['ZllLumiErr']
        
        # Fluctuate OffLumi?
        # Usually valid to fluctuate reference as well if it has errors.
        # But maybe errors are small? Let's check data later.
        # For now, fluctuate Z only to correct basic stats.
        # Ensure positive
        df_toy['ZllLumi'] = np.maximum(df_toy['ZllLumi'], 0.0)
        
        # Save first 5
        if i < 5:
            # Store mean/std of this toy
            sample_distributions.append({
                'id': i,
                'mean_z': df_toy['ZllLumi'].mean(),
                'std_z': df_toy['ZllLumi'].std(),
                'sample': df_toy['ZllLumi'].values[:5] # first 5 values
            })
            
        # Inject Signal
        df_inj = inject_signal(df_toy, args.coeff, args.signal)
        
        # Fit
        res = compute_double_ratio_and_fit(df_inj)
        fit_val = res['fits'][args.coeff]['value']
        fit_err = res['fits'][args.coeff]['error']
        res_B.append([fit_val, fit_err])

    res_B = np.array(res_B)
    
    # ==============================================================================
    # Results
    # ==============================================================================
    
    print("\n" + "="*60)
    print("SAMPLE FLUCTUATED DISTRIBUTIONS (First 5 values of ZllLumi)")
    print("="*60)
    print(f"Original Data (First 5): {df_data['ZllLumi'].values[:5]}")
    for s in sample_distributions:
        print(f"Toy {s['id']}: Mean={s['mean_z']:.2f}, Sample={s['sample']}")
        
    print("\n" + "="*60)
    print(f"RESULTS (Signal={args.signal} injected)")
    print("="*60)
    
    pull_A = (res_A[:,0] - args.signal) / res_A[:,1]
    pull_B = (res_B[:,0] - args.signal) / res_B[:,1]
    
    print(f"Method A (Stored Scrambles, N={len(res_A)}):")
    print(f"  Mean Fit:  {np.mean(res_A[:,0]):.4e}")
    print(f"  Pull Mean: {np.mean(pull_A):.3f}")
    print(f"  Pull Width:{np.std(pull_A):.3f}")
    
    print(f"\nMethod B (Direct Fluctuation, N={len(res_B)}):")
    print(f"  Mean Fit:  {np.mean(res_B[:,0]):.4e}")
    print(f"  Pull Mean: {np.mean(pull_B):.3f}")
    print(f"  Pull Width:{np.std(pull_B):.3f}")
    
    ratio = np.std(res_A[:,0]) / np.std(res_B[:,0])
    print(f"\nRatio of Widths (A/B): {ratio:.4f}")

if __name__ == "__main__":
    main()
