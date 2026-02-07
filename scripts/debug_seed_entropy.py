
import numpy as np

master_seed = 12345
ss_master = np.random.SeedSequence(master_seed)
children = ss_master.spawn(3)

print(f"Master Entropy: {ss_master.entropy}")

for i, child in enumerate(children):
    print(f"Child {i} Entropy: {child.entropy}")
    print(f"Child {i} State: {child.state}")
    
    # Check what RNG they produce
    rng_from_entropy = np.random.default_rng(child.entropy)
    print(f"  RNG(child.entropy) -> {rng_from_entropy.random()}")
    
    rng_from_obj = np.random.default_rng(child)
    print(f"  RNG(child_obj)     -> {rng_from_obj.random()}")

print("-" * 20)
print("Alternative: Generate Integer Seeds explicitly")
master_rng = np.random.default_rng(ss_master) # Or just seed
integers = master_rng.integers(0, 1000000, size=3)
for i, val in enumerate(integers):
    print(f"Int {i}: {val}")
