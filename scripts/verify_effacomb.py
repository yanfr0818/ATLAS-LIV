#!/usr/bin/env python
"""
verify_effacomb.py

Verify what ZeeEffAComb actually represents by checking relationships
between all efficiency-related columns in Set3 data.
"""

# Use pre-extracted row data to avoid slow CSV loading
# First 5 rows from data2018_shuffled_3.csv

rows = [
    # LBLive, OffLumi, ZeeRaw, ZeeN1, ZeeN2, ZeeEffTrig, ZeeErrTrig, ZeeEffReco, ZeeErrReco, 
    # ZeeEffComb, ZeeErrComb, ZeeEffAComb, ZeeErrAComb, ZeeDefTrig, ZeeDefReco, ZeeLumi, ZeeLumiErr
    {
        'LBLive': 59.245, 'OffLumi': 2.7456,
        'ZeeRaw': 63.0, 'ZeeN1': 21.0, 'ZeeN2': 42.0,
        'ZeeEffTrig': 0.8, 'ZeeErrTrig': 0.0428,
        'ZeeEffReco': 0.8541, 'ZeeErrReco': 0.0377,
        'ZeeEffComb': 0.7002, 'ZeeErrComb': 0.0630,
        'ZeeEffAComb': 0.2098, 'ZeeErrAComb': 0.0189,
        'ZeeDefTrig': 0, 'ZeeDefReco': 0,
        'ZeeLumi': 2.9211, 'ZeeLumiErr': 0.4523
    },
    {
        'LBLive': 59.586, 'OffLumi': 9.9586,
        'ZeeRaw': 234.0, 'ZeeN1': 65.0, 'ZeeN2': 169.0,
        'ZeeEffTrig': 0.8387, 'ZeeErrTrig': 0.0197,
        'ZeeEffReco': 0.8500, 'ZeeErrReco': 0.0198,
        'ZeeEffComb': 0.7036, 'ZeeErrComb': 0.0330,
        'ZeeEffAComb': 0.2108, 'ZeeErrAComb': 0.0099,
        'ZeeDefTrig': 0, 'ZeeDefReco': 0,
        'ZeeLumi': 10.7146, 'ZeeLumiErr': 0.8623
    },
    {
        'LBLive': 59.665, 'OffLumi': 16.4254,
        'ZeeRaw': 372.0, 'ZeeN1': 130.0, 'ZeeN2': 242.0,
        'ZeeEffTrig': 0.8246, 'ZeeErrTrig': 0.0175,
        'ZeeEffReco': 0.7916, 'ZeeErrReco': 0.0203,
        'ZeeEffComb': 0.6533, 'ZeeErrComb': 0.0309,
        'ZeeEffAComb': 0.1957, 'ZeeErrAComb': 0.0093,
        'ZeeDefTrig': 0, 'ZeeDefReco': 0,
        'ZeeLumi': 18.7331, 'ZeeLumiErr': 1.1697
    },
]

print("=" * 70)
print("VERIFYING ZeeEffAComb DEFINITION")
print("=" * 70)
print()

# Test 1: Is ZeeEffComb = (1 - (1-EffTrig)^2) * EffReco^2 (Equation 5)?
print("TEST 1: Verify Equation (5)")
print("  ε_comb = (1 - (1 - ε_trig)²) × ε_reco²")
print()
for i, r in enumerate(rows):
    eff_eq5 = (1 - (1 - r['ZeeEffTrig'])**2) * r['ZeeEffReco']**2
    diff = eff_eq5 - r['ZeeEffComb']
    print(f"  Row {i}: Eq5={eff_eq5:.6f}, ZeeEffComb={r['ZeeEffComb']:.6f}, Diff={diff:.6f}")

print()

# Test 2: What is ZeeEffAComb / ZeeEffComb? (should be constant if A is constant)
print("TEST 2: ZeeEffAComb / ZeeEffComb ratio")
print("  If ZeeEffAComb = A × ZeeEffComb, then ratio = A (acceptance)")
print()
for i, r in enumerate(rows):
    ratio = r['ZeeEffAComb'] / r['ZeeEffComb']
    print(f"  Row {i}: ratio = {ratio:.6f}")

ratios = [r['ZeeEffAComb'] / r['ZeeEffComb'] for r in rows]
print(f"\n  Mean ratio: {sum(ratios)/len(ratios):.6f}")
print(f"  This would be the acceptance A ≈ 0.2996")

print()

# Test 3: Check if ZeeEffAComb = const * ZeeEffComb (correlation)
print("TEST 3: Is ZeeEffAComb linearly related to ZeeEffComb?")
print()
for i, r in enumerate(rows):
    print(f"  Row {i}: EffAComb={r['ZeeEffAComb']:.4f}, EffComb={r['ZeeEffComb']:.4f}")

print()

# Test 4: Check error propagation
print("TEST 4: Error ratio (should match if A is constant)")
print("  If EffAComb = A × EffComb, then ErrAComb/ErrComb ≈ A")
print()
for i, r in enumerate(rows):
    err_ratio = r['ZeeErrAComb'] / r['ZeeErrComb']
    eff_ratio = r['ZeeEffAComb'] / r['ZeeEffComb']
    print(f"  Row {i}: ErrRatio={err_ratio:.6f}, EffRatio={eff_ratio:.6f}")

print()

# Test 5: Alternative - is EffAComb = A × (EffTrig × EffReco)?
print("TEST 5: Alternative formula ZeeEffAComb = A × EffTrig × EffReco")
print()
for i, r in enumerate(rows):
    eff_prod = r['ZeeEffTrig'] * r['ZeeEffReco']
    implied_A = r['ZeeEffAComb'] / eff_prod
    print(f"  Row {i}: EffTrig×EffReco={eff_prod:.4f}, implies A={implied_A:.4f}")

print()

# Test 6: Check the ATLAS note definition
print("=" * 70)
print("ATLAS NOTE INTERPRETATION")
print("=" * 70)
print()
print("From the ATLAS note (ATL-DAPR-PUB-2021-001):")
print()
print("  Equation (5) defines event-level efficiency:")
print("    ε_T&P = (1 - (1-ε_trig)²) × ε_reco²")
print()
print("  This matches ZeeEffComb EXACTLY (verified above)")
print()
print("  The note says ZeeLumi uses:")
print("    L = N / (σ × A_MC × ε_T&P × F_MC × t)")
print()
print("  So the 'effective' quantity seems to be:")
print("    ZeeEffAComb = A_MC × F_MC × something")
print()
print("  But ZeeEffAComb / ZeeEffComb = constant ≈ 0.30")
print("  This suggests:")
print("    ZeeEffAComb = A × ε_T&P  (i.e., F_MC is not included)")
print("  OR")
print("    ZeeEffAComb = A × F_MC × ε_T&P  (all folded together)")
print()

# Final cross-check with luminosity
print("=" * 70)
print("LUMINOSITY CROSS-CHECK")
print("=" * 70)
print()
print("Using ZeeLumi = ZeeRaw × (1-f_bkg) / (σ × ZeeEffAComb)")
print("Compute σ for different interpretations:")
print()

sigma_theory = 1.970  # nb

for i, r in enumerate(rows):
    # What sigma would make the formula work?
    sigma_implied = r['ZeeRaw'] * 0.995 / (r['ZeeLumi'] * r['ZeeEffAComb'])
    
    # If we divide by LBLive (to get rate)
    sigma_rate = r['ZeeRaw'] * 0.995 / (r['ZeeLumi'] * r['ZeeEffAComb'] * r['LBLive'])
    
    print(f"Row {i}:")
    print(f"  Implied σ (integrated): {sigma_implied:.4f} nb")
    print(f"  Implied σ (rate):       {sigma_rate:.6f} nb")
    print()

print("Compare to σ_theory = 1.970 nb")
print()
print("The implied σ ≈ 100 nb is ~50x larger than theory.")
print("This confirms ZeeLumi in Set3 is NOT the raw ATLAS formula,")
print("but has additional factors already applied.")
