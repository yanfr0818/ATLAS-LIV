import pandas as pd
import numpy as np

# Path to the data file
DATA_PATH = r"Set3\data2018_shuffled_3.csv"

def check_lblive():
    print(f"Checking LBLive in {DATA_PATH}...")
    try:
        # Load first 10 rows
        df = pd.read_csv(DATA_PATH, nrows=10)
        
        # Calculate duration
        df['Duration'] = df['LBEnd'] - df['LBStart']
        df['LiveFraction'] = df['LBLive'] / df['Duration']
        
        print("-" * 80)
        print(f"{'Row':>3} | {'LBStart':>12} | {'LBEnd':>12} | {'Duration':>8} | {'LBLive':>8} | {'Live/Dur':>8}")
        print("-" * 80)
        
        for i, row in df.iterrows():
            print(f"{i:>3} | {row['LBStart']:>12.1f} | {row['LBEnd']:>12.1f} | {row['Duration']:>8.1f} | {row['LBLive']:>8.1f} | {row['LiveFraction']:>8.1%}")
            
        print("-" * 80)
        
        # Check basic stats if possible
        print("\nSummary (First 10 rows):")
        print(f"Mean LBLive:  {df['LBLive'].mean():.2f} s")
        print(f"Mean Duration: {df['Duration'].mean():.2f} s")
        print(f"Mean Fraction: {df['LiveFraction'].mean():.1%}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {DATA_PATH}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_lblive()
