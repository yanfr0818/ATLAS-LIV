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
5. Plot Power vs Frequency (in units of f/f_sid).

Usage:
    python run_frequency_scan.py --input_file input/set3_pruned.parquet
    python run_frequency_scan.py --inject_test --test_freq 10
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm


# ---------------------------------------------------------------------------
# ATLAS Style Configuration
# ---------------------------------------------------------------------------
def set_atlas_style():
    """Apply ATLAS-style matplotlib rcParams."""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
        'font.size': 16,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',

        # Axes
        'axes.linewidth': 1.5,
        'axes.labelsize': 18,
        'axes.titlesize': 18,

        # Tick marks: inward, on all sides
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 8,
        'xtick.minor.size': 4,
        'ytick.major.size': 8,
        'ytick.minor.size': 4,
        'xtick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.major.width': 1.2,
        'ytick.minor.width': 0.8,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,

        # Legend
        'legend.frameon': False,
        'legend.fontsize': 14,

        # Figure
        'figure.figsize': (10, 7.5),
        'figure.dpi': 150,

        # No grid
        'axes.grid': False,
    })


def add_atlas_label(ax, label_type='Internal', x=0.05, y=0.95):
    """Add the 'ATLAS Internal' watermark and luminosity info."""
    ax.text(x, y, r'$\bf{ATLAS}$' + f' {label_type}',
            transform=ax.transAxes,
            fontsize=18, va='top', ha='left')
    ax.text(x, y - 0.07, r'$\sqrt{s} = 13$ TeV, 140 fb$^{-1}$',
            transform=ax.transAxes,
            fontsize=14, va='top', ha='left')

# Sidereal Frequency Constants
SIDEREAL_DAY_H = 23.9344696
SECONDS_PER_HOUR = 3600.0
SIDEREAL_DAY_S = SIDEREAL_DAY_H * SECONDS_PER_HOUR
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


def run_lomb_scargle(t, y, w, min_period_s=1.0, max_period_s=10000.0, n_freq=10000,
                     min_f_sid=None, max_f_sid=None):
    """
    Run Lomb-Scargle periodogram.
    Accepts period range (min_period_s, max_period_s) in seconds,
    or frequency range (min_f_sid, max_f_sid) in sidereal units
    for backward compatibility.
    """
    # Define frequency grid (log-spaced for even density across decades)
    t_span = t.max() - t.min()
    
    if min_f_sid is not None and max_f_sid is not None:
        # Backward compatibility: frequency in sidereal units
        f_min = min_f_sid * F_SID
        f_max = max_f_sid * F_SID
        min_period_s = 1.0 / f_max
        max_period_s = 1.0 / f_min
    else:
        f_min = 1.0 / max_period_s
        f_max = 1.0 / min_period_s
    
    print(f"Time span: {t_span/86400:.1f} days")
    print(f"Period Scan: {min_period_s:.1f} to {max_period_s:.1f} s")
    print(f"  = {f_min:.2e} Hz to {f_max:.2e} Hz")
    print(f"Grid size: {n_freq} points (log-spaced)")
    
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_freq)
    
    ls_object = None
    try:
        from astropy.timeseries import LombScargle
        print("Using Astropy Lomb-Scargle.")
        
        # Determine dy for weighting. If w is 1/sigma^2, then sigma = 1/sqrt(w)
        dy = 1.0 / np.sqrt(w + 1e-30)
        
        ls = LombScargle(t, y, dy=dy)
        ls_object = ls
        
        import time
        start_t = time.time()
        
        # 'auto' selects the best method for the grid (exact for log-spaced)
        power = ls.power(freqs, method='auto')
        
        elapsed = time.time() - start_t
        print(f"Computed {len(freqs)} frequencies in {elapsed:.2f}s.")
        
    except ImportError:
        print("Astropy not found. Please install astropy for acceptable performance over large grids.")
        print("Falling back to Numpy implementation (will be VERY slow).")
        power = lomb_scargle_numpy(t, y, freqs)
        power = power / len(t)
        
    return freqs, power, ls_object


def plot_spectrum_single(freqs, power, outdir, prefix, label=None):
    periods = 1.0 / freqs  # Convert to period in seconds
    
    fig, ax = plt.subplots()
    
    if label:
        ax.plot(periods, power, color='#1f77b4', lw=0.9, label=label, antialiased=False)
        ax.legend()
    else:
        ax.plot(periods, power, color='#1f77b4', lw=0.9, antialiased=False)
    
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Power")
    ax.set_xscale('log')
    ax.set_xlim(periods.min(), periods.max())
    ax.set_ylim(bottom=0)
    
    add_atlas_label(ax)
    
    outpath = outdir / f"frequency_spectrum_{prefix}.pdf"
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")


def plot_spectrum_multi(freqs, spectra, outdir, prefix):
    periods = 1.0 / freqs  # Convert to period in seconds
    
    fig, ax = plt.subplots()
    
    for label, power in spectra:
        ax.plot(periods, power, lw=0.9, alpha=0.8, label=str(label), antialiased=False)
    
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Power")
    ax.set_xscale('log')
    ax.set_xlim(periods.min(), periods.max())
    ax.legend()
    
    add_atlas_label(ax)
    
    outpath = outdir / f"frequency_spectrum_{prefix}.pdf"
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")


def plot_significance(freqs, power, ls_object, outdir, prefix,
                      fap_method='single'):
    """
    Plot significance (sigma) vs period.
    Left axis: sigma, right axis: p-value.
    Noise floor (sigma~0, p~1) at top, significant features dip downward.
    """
    if ls_object is None:
        print("Skipping significance plot (no LombScargle object available).")
        return
    
    periods = 1.0 / freqs  # Convert to period in seconds
    
    # Compute p-value and convert to sigma
    fap = ls_object.false_alarm_probability(power, method=fap_method)
    fap = np.clip(fap, 1e-15, 1.0)
    sigma = norm.isf(fap)  # One-sided: sigma = norm.isf(p)
    sigma = np.maximum(sigma, 0.0)  # Clip negative values (p > 0.5 gives sigma < 0)
    
    fig, ax = plt.subplots()
    
    ax.plot(periods, sigma, color='#1f77b4', lw=0.9, antialiased=False)
    
    ax.set_xlabel("Period (s)")
    ax.set_xscale('log')
    ax.set_xlim(periods.min(), periods.max())
    
    # Linear y-axis, inverted: 0 at top, max at bottom
    sigma_max = max(sigma.max() * 1.1, 3.0)
    ax.set_ylim(sigma_max, 0)  # Inverted: large at bottom, 0 at top
    ax.set_ylabel(r'Significance ($\sigma$)')
    
    # Right axis: p-value ticks at corresponding sigma positions
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())  # Inherit inverted limits
    
    # Place p-value ticks at key sigma values (consistent scientific notation)
    sigma_ticks = [0, 1, 2, 3, 4, 5]
    sigma_ticks = [s for s in sigma_ticks if s <= sigma_max]
    p_labels = [f'{norm.sf(s):.2e}' for s in sigma_ticks]
    
    ax2.set_yticks(sigma_ticks)
    ax2.set_yticklabels(p_labels)
    ax2.set_ylabel('p-value')
    ax2.tick_params(axis='y', direction='in', which='both',
                    labelsize=14, length=8)
    ax2.minorticks_off()  # Discrete p-value ticks, no minor ticks
    
    add_atlas_label(ax, x=0.05, y=0.15)
    
    outpath = outdir / f"frequency_spectrum_significance_{prefix}.pdf"
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='input/set3_pruned.parquet')
    parser.add_argument('--outdir', default='output/plots_frequency')
    parser.add_argument('--inject_test', action='store_true', help="Inject synthetic signal")
    parser.add_argument('--test_freq', type=float, default=10.0,
                        help="Injection frequency in sidereal units (default: 10)")
    parser.add_argument('--test_amp', type=float, nargs='+', default=[2e-4], help="Injection amplitude(s)")
    parser.add_argument('--min_period', type=float, default=1.0,
                        help="Min period in seconds (default: 1)")
    parser.add_argument('--max_period', type=float, default=10000.0,
                        help="Max period in seconds (default: 10000)")
    parser.add_argument('--n_freq', type=int, default=10000,
                        help="Number of frequency grid points (default: 10000)")
    parser.add_argument('--fap_method', type=str, default='single',
                        choices=['single', 'baluev', 'naive'],
                        help="FAP method: single (per-freq), baluev (trials-corrected), naive")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Apply ATLAS style globally
    set_atlas_style()
    
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
            freqs, power, ls_obj = run_lomb_scargle(t, y_inj, w,
                                                     min_period_s=args.min_period,
                                                     max_period_s=args.max_period,
                                                     n_freq=args.n_freq)
            
            amp_str = f"{amp:.1e}"
            label_prefix = f"test_inj_f{args.test_freq}_amp_{amp_str}"
            
            plot_spectrum_single(freqs, power, outdir, label_prefix)
            plot_significance(freqs, power, ls_obj, outdir, label_prefix,
                             fap_method=args.fap_method)
    else:
        freqs, power, ls_obj = run_lomb_scargle(t, y, w,
                                                 min_period_s=args.min_period,
                                                 max_period_s=args.max_period,
                                                 n_freq=args.n_freq)
        file_prefix = src_name.lower().replace(' ', '_')
        plot_spectrum_single(freqs, power, outdir, file_prefix)
        plot_significance(freqs, power, ls_obj, outdir, file_prefix,
                         fap_method=args.fap_method)
    

if __name__ == "__main__":
    main()
