
import json
import subprocess
from pathlib import Path
import sys

def main():
    results_path = Path("output/results/batch_6_results.json")
    if not results_path.exists():
        print("Results file not found!")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    master_seed = data["seed"]
    scrambles = data["scrambles"]
    
    plots_dir = "output/plots_remake_scrambles"

    for i, s in enumerate(scrambles):
        idx = s["scramble_idx"]
        child_seed = s["child_seed"]
        
        infile = f"output/scrambles_remake_plot/scramble_{idx:04d}.parquet"
        
        cmd = [
            sys.executable, "scripts/run_double_ratio.py",
            "--infile", infile,
            "--outdir", plots_dir,
            "--seed", str(master_seed),
            "--child_seed", str(child_seed),
            "--scramble_idx", str(idx),
            "--name", f"scramble_{idx}" 
        ]
        
        print(f"Plotting scramble {idx} with child_seed {child_seed}...")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
