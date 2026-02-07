import json
import subprocess
import os

# Load results
with open(r'output/results/batch_1000_results.json') as f:
    data = json.load(f)

scrambles = data['scrambles'][:5]

for s in scrambles:
    idx = s['scramble_idx']
    seed = s['child_seed']
    cmd = [
        "python", "scripts/run_double_ratio.py",
        "--infile", f"output/scrambles_pq/scramble_{idx:04d}.parquet",
        "--outdir", "output/plots_1000",
        "--name", f"scramble_{idx}",
        "--seed", "12345",
        "--scramble_idx", str(idx),
        "--child_seed", str(seed)
    ]
    print(f"Running for scramble {idx} (seed {seed})...")
    subprocess.run(cmd, check=True)

print("Done generating 5 sample plots.")
