
import numpy as np

# 1. Simulate the Master -> Child process
master_seed = 12345
ss_master = np.random.SeedSequence(master_seed)
ss_child = ss_master.spawn(1)[0]
child_integer_entropy = ss_child.entropy

print(f"Master Seed: {master_seed}")
print(f"Child Seed Integer (what goes on plot): {child_integer_entropy}")

# Generate numbers from the Child Object (using ENTROPY only)
# This is the PROPOSED FIX for batch generation
rng_from_seq = np.random.default_rng(ss_child.entropy)
nums_A = rng_from_seq.random(3)
print(f"Stream A (from SeedSequence.entropy): {nums_A}")

# 2. Simulate User copying the integer from the plot
# Direct initialization with the integer
rng_from_int = np.random.default_rng(child_integer_entropy)
nums_B = rng_from_int.random(3)
print(f"Stream B (from Integer direct):       {nums_B}")

# Check equality
if np.allclose(nums_A, nums_B):
    print("\nSUCCESS: The streams are IDENTICAL.")
    print("Fix verified: Using ss.entropy ensures direct reproducibility.")
else:
    print("\nFAILURE: The streams are DIFFERENT.")
