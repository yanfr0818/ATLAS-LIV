#!/usr/bin/env python
"""
sme_fit.py

SME (Standard Model Extension) coefficient fitting for LIV analysis.
Fits the double ratio to SME operator templates: RD(phi) = 1 + c * f(phi)

Based on the tag11_double_ratio.py implementation from Enrico's work.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

# 12 SME coefficients (4 per quark type: u, c, d)
SME_COEFFICIENTS = (
    "duXZ", "duYZ", "duXXYY", "duXY",
    "cuXZ", "cuYZ", "cuXXYY", "cuXY",
    "cdXZ", "cdYZ", "cdXXYY", "cdXY",
)

# Display exponents for scaling coefficients to O(1) values
SME_DISPLAY_EXPONENTS = (6, 6, 6, 7, 6, 6, 6, 7, 4, 4, 5, 5)


def sme_template(name: str, phi: np.ndarray) -> np.ndarray:
    """
    SME operator template function f(name, phi).
    
    The linear model is: RD(phi) ≈ 1 + c * f(phi)
    where c is the SME coefficient being fitted.
    
    These templates are derived from the physics of LIV and depend on
    the detector's latitude, orientation, and the specific SME operator.
    Coefficients are from Enrico's Mathematica notebook (tag11.nb In[59]).
    
    Args:
        name: SME coefficient name (e.g., 'duXZ', 'cuYZ')
        phi: Phase values in [0, 1)
        
    Returns:
        Template function evaluated at each phi
    """
    phi = np.asarray(phi, dtype=float)
    x2 = 2.0 * np.pi * phi  # 2π * φ for period-2 terms
    x4 = 4.0 * np.pi * phi  # 4π * φ for period-4 terms
    
    # du (up quark d-type coefficients)
    if name == "duXZ":
        return 6.28069 * np.cos(x2) - 41.0569 * np.sin(x2)
    if name == "duYZ":
        return 41.0569 * np.cos(x2) + 6.28069 * np.sin(x2)
    if name == "duXXYY":
        return 77.6067 * np.cos(x4) + 24.3128 * np.sin(x4)
    if name == "duXY":
        return -48.6256 * np.cos(x4) + 155.213 * np.sin(x4)
    
    # cu (up quark c-type coefficients)
    if name == "cuXZ":
        return 8.084 * np.cos(x2) - 52.8451 * np.sin(x2)
    if name == "cuYZ":
        return 52.8451 * np.cos(x2) + 8.084 * np.sin(x2)
    if name == "cuXXYY":
        return 99.8891 * np.cos(x4) + 31.2935 * np.sin(x4)
    if name == "cuXY":
        return -62.5869 * np.cos(x4) + 199.778 * np.sin(x4)
    
    # cd (down quark c-type coefficients)
    if name == "cdXZ":
        return 0.181551 * np.cos(x2) - 1.1868 * np.sin(x2)
    if name == "cdYZ":
        return 1.1868 * np.cos(x2) + 0.181551 * np.sin(x2)
    if name == "cdXXYY":
        return 2.24331 * np.cos(x4) + 0.702788 * np.sin(x4)
    if name == "cdXY":
        return -1.40558 * np.cos(x4) + 4.48662 * np.sin(x4)
    
    raise KeyError(f"Unknown SME coefficient '{name}'")


@dataclass(frozen=True)
class FitResult:
    """Result of fitting a single SME coefficient."""
    coeff: str      # Coefficient name
    value: float    # Fitted value
    error: float    # Standard error
    
    @property
    def significance(self) -> float:
        """Significance in units of sigma (|value/error|)."""
        if self.error > 0 and np.isfinite(self.error):
            return abs(self.value / self.error)
        return np.nan


def fit_single_sme_coeff(
    phi: np.ndarray,
    rd: np.ndarray,
    rd_err: np.ndarray,
    coeff_name: str,
    drop_last_bin: bool = True,
    eps: float = 1e-12,
) -> FitResult:
    """
    Weighted least-squares fit for a single SME coefficient.
    
    Fits the model: RD(phi) = 1 + c * f(phi)
    where f(phi) is the SME template for the given coefficient.
    
    Args:
        phi: Phase bin centers [0, 1)
        rd: Double ratio values per bin
        rd_err: Errors on double ratio per bin
        coeff_name: Name of SME coefficient to fit
        drop_last_bin: If True, drop the last valid bin (notebook convention)
        eps: Small number for numerical stability
        
    Returns:
        FitResult with fitted value and error
    """
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(rd, dtype=float)
    s = np.asarray(rd_err, dtype=float)
    
    # Valid bins: finite values with positive errors
    mask = np.isfinite(y) & np.isfinite(s) & (s > eps)
    
    if drop_last_bin and mask.sum() > 1:
        # Remove last valid bin (notebook convention to avoid phase wrapping issues)
        last = np.where(mask)[0][-1]
        mask[last] = False
    
    if mask.sum() < 1:
        return FitResult(coeff=coeff_name, value=np.nan, error=np.nan)
    
    phi_use = phi[mask]
    y_use = y[mask]
    s_use = s[mask]
    f_use = sme_template(coeff_name, phi_use)
    
    # Weighted least squares: minimize sum(w * (y - 1 - c*f)^2)
    # Optimal c = sum(w * f * (y-1)) / sum(w * f^2)
    w = 1.0 / (s_use * s_use)
    denom = np.sum(w * f_use * f_use)
    
    if denom <= eps:
        return FitResult(coeff=coeff_name, value=np.nan, error=np.nan)
    
    num = np.sum(w * f_use * (y_use - 1.0))
    c_hat = num / denom
    c_err = np.sqrt(1.0 / denom)
    
    return FitResult(coeff=coeff_name, value=float(c_hat), error=float(c_err))


def fit_all_sme_coeffs(
    phi: np.ndarray,
    rd: np.ndarray,
    rd_err: np.ndarray,
    coeffs: Tuple[str, ...] = SME_COEFFICIENTS,
    drop_last_bin: bool = True,
) -> Dict[str, FitResult]:
    """
    Fit all SME coefficients to the double ratio.
    
    Args:
        phi: Phase bin centers [0, 1)
        rd: Double ratio values per bin
        rd_err: Errors on double ratio per bin
        coeffs: Tuple of coefficient names to fit
        drop_last_bin: If True, drop the last valid bin
        
    Returns:
        Dict mapping coefficient name to FitResult
    """
    return {
        c: fit_single_sme_coeff(phi, rd, rd_err, c, drop_last_bin=drop_last_bin)
        for c in coeffs
    }


def fit_results_to_dict(results: Dict[str, FitResult]) -> Dict[str, Dict]:
    """Convert fit results to a JSON-serializable dict."""
    return {
        name: {
            "value": r.value,
            "error": r.error,
            "significance": r.significance,
        }
        for name, r in results.items()
    }


def print_fit_summary(results: Dict[str, FitResult], title: str = "SME Fit Results"):
    """Print a formatted table of fit results."""
    print(f"\n{title}")
    print("=" * 50)
    print(f"{'Coeff':<10} {'Value':>14} {'Error':>14} {'Signif':>8}")
    print("-" * 50)
    for name, r in results.items():
        if np.isfinite(r.value):
            print(f"{name:<10} {r.value:>14.6e} {r.error:>14.6e} {r.significance:>8.2f}σ")
        else:
            print(f"{name:<10} {'NaN':>14} {'NaN':>14} {'NaN':>8}")
    print("=" * 50)


# ============================================================
# Test/demo usage
# ============================================================
if __name__ == "__main__":
    # Generate synthetic data to test fitting
    np.random.seed(42)
    
    nbins = 100
    phi = (np.arange(nbins) + 0.5) / nbins
    
    # Synthetic double ratio: RD = 1 + noise
    rd = 1.0 + np.random.normal(0, 0.002, nbins)
    rd_err = np.full(nbins, 0.002)
    
    # Fit all coefficients
    results = fit_all_sme_coeffs(phi, rd, rd_err)
    print_fit_summary(results, "Test Fit (null signal)")
