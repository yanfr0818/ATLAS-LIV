#!/usr/bin/env python
"""
run_frequency_scan.py

Performs an unbinned frequency spectrum analysis on the ATLAS LIV data
using the Lomb-Scargle periodogram.

Methodology:
1. Load Set3 data (parquet).
2. Filter for valid LBs (OffLumi > 0, ZllLumi > 0).
3. Construct Time Series:
   - t: Central time of LB
   - y: Normalized Double Ratio (Zll/Off / <Zll/Off> - 1)
   - w: 1/err^2
4. Compute Lomb-Scargle Power Spectrum over a specified frequency range.
5. Plot Power vs Frequency (in units of Sidereal Frequency).

Usage:
    python run_frequency_scan.py --input_file input/set3_pruned.parquet
    python run_frequency_scan.py --inject_test --test_freq 3.5
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Sidereal Frequency Constants
SIDEREAL_DAY_H = 23.9344696
SECONDS_PER_HOUR = 3600.0
SIDEREAL_DAY_S = SIDEREAL_DAY_H * SECONDS_PER_HOUR
OMEGA_SID = 2 * np.pi / SIDEREAL_DAY_S
F_SID = 1.0 / SIDEREAL_DAY_S  # Hz (~1.16e-5 Hz)


def load_data(input_file: str):
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows.")
    
    # Filter valid LBs
    mask = (df['OffLumi'] > 0) & (df['ZllLumi'] > 0)
    df = df[mask].reset_index(drop=True)
    print(f"Valid LBs: {len(df)}")
    return df


def prepare_timeseries(df: pd.DataFrame):
    """
    Extract t, y, w from DataFrame.
    t: Time in seconds
    y: Normalized deviation (RD - 1)
    w: Weights (1/sigma^2)
    """
    # Time
    t_start = df['LBStart'].values
    t_end = df['LBEnd'].values
    t = 0.5 * (t_start + t_end)
    
    # Sort by time just in case
    idx = np.argsort(t)
    t = t[idx]
    
    # Signal
    z = df['ZllLumi'].values[idx]
    z_err = df['ZllLumiErr'].values[idx]
    off = df['OffLumi'].values[idx]
    
    ratio = z / off
    mean_ratio = np.sum(z) / np.sum(off) # Global mean
    
    y = ratio / mean_ratio - 1.0
    
    # Errors
    # sigma_r = r * (sigma_z / z)
    # sigma_y = sigma_r / mean_ratio = (r/mean_ratio) * (sigma_z/z) = (y+1) * (sigma_z/z)
    # Approx: sigma_y ~= 1 * (sigma_z/z) since y is small.
    # Let's use the exact propagation:
    sigma_ratio = ratio * (z_err / z)
    sigma_y = sigma_ratio / mean_ratio
    
    w = 1.0 / (sigma_y**2)
    
    return t, y, w


def inject_signal(t: np.ndarray, y: np.ndarray, freq_sid: float, amp: float = 1e-3, seed: int = 42):
    """
    Inject a sinusoidal signal: A * cos(2*pi*f*t + phase)
    freq_sid: Frequency in units of F_SID
    """
    print(f"Injecting signal: Freq={freq_sid}*f_sid, Amp={amp:.1e}")
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2*np.pi)
    
    f_inj = freq_sid * F_SID
    signal = amp * np.cos(2*np.pi*f_inj*t + phase)
    
    y_new = y + signal
    return y_new


def lomb_scargle_numpy(t, y, freqs):
    """
    Compute Lomb-Scargle periodogram using pure numpy.
    P(f) = 0.5 * ( (sum(y*cos)^2 / sum(cos^2)) + (sum(y*sin)^2 / sum(sin^2)) )
    using the phase shift tau.
    """
    omegas = 2 * np.pi * freqs
    power = np.zeros_like(freqs)
    
    chunk_size = 100
    n_freqs = len(freqs)
    
    y = y - np.mean(y)
    
    import time
    start_t = time.time()
    for i in range(0, n_freqs, chunk_size):
        end = min(i + chunk_size, n_freqs)
        w_chunk = omegas[i:end]  # Shape (M,)
        
        wt = np.outer(w_chunk, t)
        
        sin2wt = np.sin(2*wt)
        cos2wt = np.cos(2*wt)
        
        ss2 = np.sum(sin2wt, axis=1)
        cc2 = np.sum(cos2wt, axis=1)
        tau = 0.5 * np.arctan2(ss2, cc2)
        
        wt_tau = wt - tau[:, None]
        
        cos_wt_tau = np.cos(wt_tau)
        sin_wt_tau = np.sin(wt_tau)
        
        sum_cos2 = np.sum(cos_wt_tau**2, axis=1)
        sum_sin2 = np.sum(sin_wt_tau**2, axis=1)
        
        sum_y_cos = np.dot(cos_wt_tau, y)
        sum_y_sin = np.dot(sin_wt_tau, y)
        
        p = 0.5 * ( (sum_y_cos**2 / sum_cos2) + (sum_y_sin**2 / sum_sin2) )
        power[i:end] = p
        
        if (i // chunk_size) % 10 == 0:
            elapsed = time.time() - start_t
            rate = (i + chunk_size) / elapsed if elapsed > 0 else 0
            eta = (n_freqs - i) / rate if rate > 0 else 0
            print(f"Processed {min(i + chunk_size, n_freqs)}/{n_freqs} freqs ({rate:.1f}/s), ETA {eta:.1f}s", flush=True)
            
    return power


def run_lomb_scargle(t, y, w, min_f_sid=0.1, max_f_sid=10.0, n_freq=10000):
    """
    Run Lomb-Scargle periodogram.
    Range: [min_f_sid, max_f_sid] in units of sidereal frequency.
    """
    # Define frequency grid
    t_span = t.max() - t.min()
    
    f_min = min_f_sid * F_SID
    f_max = max_f_sid * F_SID
    df = (f_max - f_min) / n_freq
    
    print(f"Time span: {t_span/86400:.1f} days")
    print(f"Frequency Scan: {f_min:.2e} Hz to {f_max:.2e} Hz")
    print(f"Resolution approx: {df:.2e} Hz")
    print(f"Grid size: {n_freq} points")
    
    freqs = np.linspace(f_min, f_max, n_freq)
    


    try:
        from astropy.timeseries import LombScargle
        print("Using Astropy Lomb-Scargle (O(N log N) fast method).")
        # astropy takes cyclic frequency rather than angular frequency.
        # freqs array is already cyclic (f)
        
        # Determine dy for weighting. If w is 1/sigma^2, then sigma = 1/sqrt(w)
        # Note: if w=0 or inf, we'd need to clean it. Let's ensure valid weights.
        dy = 1.0 / np.sqrt(w + 1e-30)
        
        # LombScargle class structure
        ls = LombScargle(t, y, dy=dy)
        
        import time
        start_t = time.time()
        
        # 'fast' method uses O(N log N) extirpation and FFT.
        power = ls.power(freqs, method='fast')
        
        elapsed = time.time() - start_t
        print(f"Computed {len(freqs)} frequencies in {elapsed:.2f}s using 'fast' algorithm.")
        
    except ImportError:
        print("Astropy not found. Please install astropy for acceptable performance over large grids.")
        print("Falling back to Numpy implementation (will be VERY slow).")
        power = lomb_scargle_numpy(t, y, freqs)
        power = power / len(t)
        print("Scipy not found. Using Custom Numpy Lomb-Scargle.")
        power = lomb_scargle_numpy(t, y, freqs)
        power = power / len(t)
        
    return freqs, power


def plot_spectrum_single(freqs, power, outdir, prefix, max_f_sid, title_prefix, label=None, zoom_center=None, zoom_width=2.0):
    # Convert x-axis to Sidereal Frequencies
    freq_sid_units = freqs / F_SID
    
    def _make_plot(suffix="", xlim=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot power
        if label:
            ax.plot(freq_sid_units, power, color='cornflowerblue', lw=0.8, label=label)
            ax.legend()
        else:
            ax.plot(freq_sid_units, power, color='cornflowerblue', lw=0.8)
            
        # Annotate integer harmonics
        for i in range(1, int(max_f_sid) + 1):
            if i > 10 and xlim is None: break # Only mark low harmonics on full plot
            if xlim and not (xlim[0] <= i <= xlim[1]): continue
            ax.axvline(i, color='r', alpha=0.3, ls='--', lw=0.8)
            
        ax.set_xlabel(r"Frequency ($f / f_{sid}$)")
        ax.set_ylabel("Power")
        ax.set_title(f"Lomb-Scargle Periodogram - {title_prefix}")
        
        if xlim:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(max(0, freq_sid_units.min()), freq_sid_units.max())
            
        ax.grid(True, alpha=0.3)
        
        outpath = outdir / f"frequency_spectrum_{prefix}{suffix}.pdf"
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        print(f"Saved {outpath}")

    # Full plot
    _make_plot()
    
    # Zoomed plot
    if zoom_center is not None:
        _make_plot(suffix="_zoomed", xlim=(zoom_center - zoom_width/2, zoom_center + zoom_width/2))


def plot_spectrum_multi(freqs, spectra, outdir, prefix, max_f_sid, title_prefix, zoom_center=None, zoom_width=2.0):
    # Convert x-axis to Sidereal Frequencies
    freq_sid_units = freqs / F_SID
    
    def _make_plot(suffix="", xlim=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for label, power in spectra:
            ax.plot(freq_sid_units, power, lw=0.8, alpha=0.8, label=str(label))
        
        # Annotate integer harmonics
        for i in range(1, int(max_f_sid) + 1):
            if i > 10 and xlim is None: break # Only mark low harmonics on full plot
            if xlim and not (xlim[0] <= i <= xlim[1]): continue
            ax.axvline(i, color='cornflowerblue', alpha=0.5, ls='--', lw=0.8)
            
        ax.set_xlabel(r"Frequency ($f / f_{sid}$)")
        ax.set_ylabel("Power")
        ax.set_title(f"Lomb-Scargle Periodogram - {title_prefix}")
        
        if xlim:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(freq_sid_units.min(), freq_sid_units.max())
            
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        outpath = outdir / f"frequency_spectrum_{prefix}{suffix}.pdf"
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        print(f"Saved {outpath}")

    # Full plot
    _make_plot()
    
    # Zoomed plot
    if zoom_center is not None:
        _make_plot(suffix="_zoomed", xlim=(zoom_center - zoom_width/2, zoom_center + zoom_width/2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='input/set3_pruned.parquet')
    parser.add_argument('--outdir', default='output/plots_frequency')
    parser.add_argument('--inject_test', action='store_true', help="Inject synthetic signal")
    parser.add_argument('--test_freq', type=float, default=3.5, help="Injection frequency (sidereal units)")
    parser.add_argument('--test_amp', type=float, nargs='+', default=[2e-4], help="Injection amplitude(s)")
    parser.add_argument('--max_freq', type=float, default=10.0, help="Max freq in sidereal units")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    if args.input_file and Path(args.input_file).exists():
        df = load_data(args.input_file)
    else:
        if args.inject_test or 'scramble' in args.input_file:
            print("Generating synthetic background/dummy data if file missing...")
            raise FileNotFoundError(f"{args.input_file} not found.")
            
    # Prepare Time Series
    t, y, w = prepare_timeseries(df)
    
    # Determine data source name for title
    src_name = Path(args.input_file).stem
    if src_name == 'set3_pruned':
        src_name = 'Original Data'
    else:
        src_name = src_name.replace('_', ' ').title()

    # Analyze and Plot
    if args.inject_test:
        print("--- TEST MODE ---")
        for amp in args.test_amp:
            y_inj = inject_signal(t, y, args.test_freq, amp)
            freqs, power = run_lomb_scargle(t, y_inj, w, max_f_sid=args.max_freq)
            
            amp_str = f"{amp:.1e}"
            label_prefix = f"test_inj_{args.test_freq}_amp_{amp_str}"
            title = f"{src_name} (Injected: Freq={args.test_freq} f_sid, Amp={amp_str})"
            
            plot_spectrum_single(freqs, power, outdir, label_prefix, args.max_freq, title)
    else:
        freqs, power = run_lomb_scargle(t, y, w, max_f_sid=args.max_freq)
        file_prefix = src_name.lower().replace(' ', '_')
        plot_spectrum_single(freqs, power, outdir, file_prefix, args.max_freq, src_name)
    

if __name__ == "__main__":
    main()
