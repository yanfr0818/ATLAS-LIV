#!/usr/bin/env python
"""
compare_stats_methods.py

Comparision of Scramble vs Poisson Injection using Binned (LumiBlock) toy data.
"""
import numpy as np
import argparse
import time

def weighted_linear_regression(x, y, w):
    """
    Fit y = beta0 + beta1 * x
    """
    sum_w = np.sum(w)
    sum_wx = np.sum(w * x)
    sum_wy = np.sum(w * y)
    sum_wxx = np.sum(w * x * x)
    sum_wxy = np.sum(w * x * y)
    
    denom = sum_w * sum_wxx - sum_wx * sum_wx
    if denom == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    beta0 = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denom
    beta1 = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
    
    var_beta0 = sum_wxx / denom
    var_beta1 = sum_w / denom
    
    return beta0, beta1, np.sqrt(var_beta0), np.sqrt(var_beta1)

def fit_double_ratio(t_vals, n_sig, n_ref, n_phase_bins=100):
    """
    Fit Ratio = N_sig / N_ref = K * (1 + A * cos(t))
    Using binned weighted regression.
    """
    phase = np.mod(t_vals, 2*np.pi)
    bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)
    idx = np.digitize(phase, bins) - 1
    idx = np.clip(idx, 0, n_phase_bins - 1)
    
    bin_sig = np.zeros(n_phase_bins)
    bin_ref = np.zeros(n_phase_bins)
    
    np.add.at(bin_sig, idx, n_sig)
    np.add.at(bin_ref, idx, n_ref)
    
    # Calculate Ratio R = S/R
    # Error on Ratio: dR/R = sqrt(1/S + 1/R) (Assuming Poisson)
    # Actually dR = R * sqrt(1/S + 1/R)
    
    mask = (bin_ref > 0) & (bin_sig > 0)
    
    y = np.zeros(n_phase_bins)
    y[mask] = bin_sig[mask] / bin_ref[mask]
    
    y_err = np.zeros(n_phase_bins)
    # var(R) = R^2 * (1/S + 1/Ref)
    y_err[mask] = y[mask] * np.sqrt(1.0/bin_sig[mask] + 1.0/bin_ref[mask])
    
    # Fit y = C * (1 + A * x)
    # y = C + (C*A) * x
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    x = np.cos(bin_centers)
    
    w = np.zeros_like(y)
    w[mask] = 1.0 / (y_err[mask]**2)
    
    b0, b1, b0_err, b1_err = weighted_linear_regression(x[mask], y[mask], w[mask])
    
    if b0 == 0:
        return 0.0, 0.0
        
    A = b1 / b0
    A_err = b1_err / b0
    
    return A, A_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_toys', type=int, default=10000)
    parser.add_argument('--n_lbs', type=int, default=2000)
    parser.add_argument('--avg_rate', type=float, default=500.0) 
    parser.add_argument('--signal', type=float, default=1e-5)
    args = parser.parse_args()
    
    print(f"Comparing DR Methods: N_toys={args.n_toys}, Rate={args.avg_rate}, Sig={args.signal} (Varying Lumi)")
    
    # Lumi Profile
    t_lbs = np.linspace(0, 2*np.pi, args.n_lbs)
    base_profile = np.linspace(1.5, 0.5, args.n_lbs)
    jitter = np.random.normal(1.0, 0.2, args.n_lbs)
    jitter = np.clip(jitter, 0.5, 1.5)
    lumi_profile = args.avg_rate * base_profile * jitter
    lumi_profile = lumi_profile * (args.avg_rate / np.mean(lumi_profile))
    
    print(f"Lumi Profile: Mean={np.mean(lumi_profile):.1f}, Var={np.var(lumi_profile):.1f}")
    
    # Generate PARENT DATA (Observed)
    # Two channels: Signal and Reference
    # Both follow Lumi Profile. Signal has intrinsic modulation? No, parent is null.
    np.random.seed(42)
    obs_sig = np.random.poisson(lumi_profile).astype(float)
    obs_ref = np.random.poisson(lumi_profile).astype(float) # Assuming ref has same rate for simplicity
    
    obs_mean_sig = np.mean(obs_sig)
    obs_mean_ref = np.mean(obs_ref)
    
    res_A = []
    res_B = []
    
    start = time.time()
    for i in range(args.n_toys):
        if (i+1) % 1000 == 0:
            print(f"Toy {i+1}...")
            
        # ==========================================================
        # Method A: Scramble (Permute observed PAIRS)
        # ==========================================================
        # We assume Signal and Ref are coupled per LB (same time/conditions).
        # We scramble the (Sig, Ref) pair together relative to time t.
        
        perm = np.random.permutation(len(obs_sig))
        sig_scr = obs_sig[perm]
        ref_scr = obs_ref[perm]
        
        # Inject Signal into Signal Channel Only
        # N_sig' = N_sig * (1 + A cos t)
        mod = 1.0 + args.signal * np.cos(t_lbs)
        sig_A = sig_scr * mod
        ref_A = ref_scr # Reference unaffected
        
        fit_A, err_A = fit_double_ratio(t_lbs, sig_A, ref_A)
        res_A.append([fit_A, err_A])
        
        # ==========================================================
        # Method B: Poisson Bootstrap (Resample PAIRS)
        # ==========================================================
        # Sample LBs (pairs of rates) with replacement
        idx_boot = np.random.randint(0, len(obs_sig), len(obs_sig))
        
        rate_boot_sig = obs_sig[idx_boot]
        rate_boot_ref = obs_ref[idx_boot]
        
        # Modulate Signal Rate
        rate_mod_sig = rate_boot_sig * (1.0 + args.signal * np.cos(t_lbs))
        
        # Generate Poisson
        sig_B = np.random.poisson(rate_mod_sig).astype(float)
        ref_B = np.random.poisson(rate_boot_ref).astype(float)
        
        fit_B, err_B = fit_double_ratio(t_lbs, sig_B, ref_B)
        res_B.append([fit_B, err_B])

    res_A = np.array(res_A)
    res_B = np.array(res_B)
    
    pull_A = (res_A[:,0] - args.signal) / res_A[:,1]
    pull_B = (res_B[:,0] - args.signal) / res_B[:,1]
    
    print("\nRESULTS:")
    print(f"Method A (Scramble): MeanFit={np.mean(res_A[:,0]):.4e}, PullMean={np.mean(pull_A):.3f}, PullWidth={np.std(pull_A):.3f}")
    print(f"Method B (Poisson):  MeanFit={np.mean(res_B[:,0]):.4e}, PullMean={np.mean(pull_B):.3f}, PullWidth={np.std(pull_B):.3f}")
    
    ratio = np.std(res_A[:,0]) / np.std(res_B[:,0])
    print(f"Ratio of Fit Widths (A/B): {ratio:.4f}")

if __name__ == "__main__":
    main()
