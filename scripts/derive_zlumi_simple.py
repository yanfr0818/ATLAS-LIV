#!/usr/bin/env python
"""
More detailed analysis - check if ZeeLumi formula includes efficiency corrections
"""
import numpy as np

# More rows for better statistics (from data2018_shuffled_3.csv)
# Format: (ZeeRaw, ZeeN1, ZeeN2, ZeeEffTrig, ZeeEffReco, ZeeEffComb, ZeeEffAComb, ZeeLumi, ZeeLumiErr)
rows = [
    (63.0, 21.0, 42.0, 0.8, 0.8541, 0.7002, 0.2098, 2.9211, 0.4523),
    (234.0, 65.0, 169.0, 0.8387, 0.8500, 0.7036, 0.2108, 10.7146, 0.8623),
    (372.0, 130.0, 242.0, 0.8246, 0.7916, 0.6533, 0.1953, 18.7331, 1.1697),
]

print("=== Detailed ZeeLumi Formula Analysis ===\n")

# The standard ATLAS Z-counting formula is:
# L = N / (sigma_fid * C)
# where C is the acceptance x efficiency correction factor
# sigma_fid for Z->ll is about 2 nb (combined ee+mumu)

# Let's check if the formula uses EffAComb (which includes acceptance)
# ZeeLumi = ZeeRaw / (sigma_fid * ZeeEffAComb)

print("Testing: ZeeLumi_derived = ZeeRaw / (sigma * EffAComb)")
print("Finding best sigma...")

# The "sigma" we derive should be the fiducial cross section
# Typical Z->ee fiducial is ~0.5 nb (electron channel only, fiducial region)

# From the first row:
# ZeeLumi = 2.9211 nb^-1
# ZeeRaw = 63 events  
# ZeeEffAComb = 0.2098

# If ZeeLumi = ZeeRaw / (sigma * eff)
# Then sigma = ZeeRaw / (ZeeLumi * eff) = 63 / (2.9211 * 0.2098) = 102.8 nb

# But wait - the efficiencies in Set3 might already be factored in!
# Let me check if ZeeLumi = ZeeRaw / sigma_effective where sigma_effective is constant

print()
print("=== Hypothesis: ZeeLumi already includes efficiency correction ===")
print("Formula: ZeeLumi = (ZeeRaw / ZeeEffComb) / sigma_fid_raw")
print("         = ZeeRaw / (ZeeEffComb * sigma_fid_raw)")
print()

for r in rows:
    zee_raw, n1, n2, eff_trig, eff_reco, eff_comb, eff_acomb, zee_lumi, _ = r
    
    # Test: efficiency-corrected raw count
    zee_corrected = zee_raw / eff_comb
    sigma_with_corr = zee_corrected / zee_lumi
    
    # Or: ZeeLumi = ZeeRaw / (sigma * EffAComb)
    sigma_with_acomb = zee_raw / (zee_lumi * eff_acomb)
    
    print(f"ZeeRaw={zee_raw:.0f}, EffComb={eff_comb:.4f}, EffAComb={eff_acomb:.4f}")
    print(f"  sigma (from corrected) = {sigma_with_corr:.4f}")
    print(f"  sigma (from EffAComb) = {sigma_with_acomb:.4f}")
    print()

# Let's try the most consistent formula
print("=== Testing with EffAComb ===")
# Using the mean sigma from EffAComb derivation
sigmas = [r[0] / (r[7] * r[6]) for r in rows]
sigma_mean = np.mean(sigmas)
print(f"Mean sigma_fid (with EffAComb): {sigma_mean:.4f} nb")

print("\nValidation:")
for r in rows:
    derived = r[0] / (sigma_mean * r[6])
    actual = r[7]
    pct = (derived - actual) / actual * 100
    print(f"  Raw={r[0]:.0f}, EffAComb={r[6]:.4f} -> Derived: {derived:.4f}, Actual: {actual:.4f}, Diff: {pct:+.2f}%")

print()
print("=== Testing with EffComb ===")
sigmas_comb = [r[0] / (r[7] * r[5]) for r in rows]
sigma_comb_mean = np.mean(sigmas_comb)
print(f"Mean sigma_fid (with EffComb): {sigma_comb_mean:.4f}")

print("\nValidation:")
for r in rows:
    derived = r[0] / (sigma_comb_mean * r[5])
    actual = r[7]
    pct = (derived - actual) / actual * 100
    print(f"  Raw={r[0]:.0f}, EffComb={r[5]:.4f} -> Derived: {derived:.4f}, Actual: {actual:.4f}, Diff: {pct:+.2f}%")

print()
print("=== Final Answer ===")
print(f"Best formula: ZeeLumi = ZeeRaw / ({sigma_comb_mean:.4f} * ZeeEffComb)")
print(f"Or equivalently: ZeeLumi = ZeeRaw / ({sigma_mean:.4f} * ZeeEffAComb)")
print()
print(f"The sigma_fid * acceptance factor = {sigma_comb_mean:.4f} pb (with EffComb)")
print(f"The sigma_fid (full) = {sigma_mean:.4f} pb (with EffAComb)")
print()
print("Note: This makes sense if:")
print(f"  - sigma_fid (electron channel) ~ {sigma_mean/1000:.3f} fb (typ. 0.5-1 nb)")
print(f"  - Acceptance ~ {np.mean([r[6]/r[5] for r in rows]):.4f}")
