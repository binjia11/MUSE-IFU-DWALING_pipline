"""Shared constants and helpers for the MUSE IFU pipeline.

Paths are configured via environment variables (set by the driver script):
  PIPE_CUBE_PATH  – full path to the input MUSE cube
  PIPE_OUT_DIR    – output directory for this galaxy
  PIPE_V_SYS      – systemic velocity in km/s (default 0)
"""
import os
import numpy as np
from astropy.io import fits

ROOT = "/Users/binjia/Desktop/low-metallicity_shocks_LMC"

# --- configurable paths (env vars with sensible defaults) ---
CUBE_PATH = os.environ.get("PIPE_CUBE_PATH", os.path.join(ROOT, "HEN_2-10.fits"))
OUT_DIR   = os.environ.get("PIPE_OUT_DIR",   os.path.join(ROOT, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

GALAXY = os.environ.get("PIPE_GALAXY", os.path.basename(CUBE_PATH).replace(".fits", ""))

V_SYS = float(os.environ.get("PIPE_V_SYS", "0.0"))

C_KMS   = 299792.458
Z_SYS   = V_SYS / C_KMS

# Cresci 2017 line list (rest-frame air wavelengths in Å)
LINES = {
    "Hbeta":       4861.333,
    "OIII4959":    4958.911,
    "OIII5007":    5006.843,
    "OI6300":      6300.304,
    "NII6548":     6548.050,
    "Halpha":      6562.819,
    "NII6584":     6583.460,
    "HeI6678":     6678.151,
    "SII6716":     6716.440,
    "SII6731":     6730.815,
    "HeI7065":     7065.196,
    "SIII9069":    9068.600,
}

# Fixed-ratio doublets (intensity_ratio = bright/faint, from atomic physics)
DOUBLET_RATIOS = {
    ("OIII5007", "OIII4959"): 2.98,
    ("NII6584", "NII6548"):    2.94,
}

# Continuum reference window for Voronoi S/N (rest-frame Å)
CONT_REST_RANGE = (5300.0, 5530.0)

# Stellar fit wavelength range (observed-frame left edge, rest-frame right edge)
FIT_REST_RANGE = (4760.0, 7400.0)

# Local refinement window around emission lines (rest-frame Å half-width)
LOCAL_WIN_HW  = 60.0
LINE_MASK_HW  = 12.0

# pPXF / Voronoi settings
TARGET_SN_BIN     = 50.0
SN_FLOOR_VORONOI  = 1.0

# Detection threshold
SN_DETECT = 3.0


def muse_lsf_fwhm(lam_aa):
    """MUSE LSF FWHM (Å) from Bacon et al. 2017 polynomial fit."""
    return 5.835e-8 * lam_aa ** 2 - 9.080e-4 * lam_aa + 5.983


def measure_systemic_velocity(cube_path=None, verbose=True):
    """Measure systemic velocity by cross-correlating with Halpha+[NII] template.

    Sums the spectrum over bright spaxels, then cross-correlates with a
    template of the Hα + [NII]6548,6584 complex to find the best-matching
    observed wavelength.

    Parameters
    ----------
    cube_path : str or None
        Path to the MUSE cube. Uses CUBE_PATH from env if None.
    verbose : bool
        Print measurements.

    Returns
    -------
    v_sys : float
        Systemic velocity in km/s.
    """
    data, var, wave, hdr0, hdr1, hdul = load_cube(cube_path, memmap=True)
    ha_rest = 6562.819

    # Use central 20% of the field for an integrated spectrum
    ny, nx = data.shape[1], data.shape[2]
    y0, y1 = int(ny * 0.4), int(ny * 0.6)
    x0, x1 = int(nx * 0.4), int(nx * 0.6)
    central = np.nansum(data[:, y0:y1, x0:x1], axis=(1, 2))

    # Build a simple Hα + [NII] template (Gaussians at rest wavelengths)
    # Hα 6562.8 + [NII]6548 + [NII]6584
    lines = [6548.050, 6562.819, 6583.460]
    amps = [0.33, 1.0, 0.33]  # approximate relative amplitudes
    sigma_aa = 3.0  # Gaussian sigma in Å (typical for MUSE)

    # Search over a grid of trial velocities
    v_trials = np.linspace(-2000, 8000, 1000)  # km/s
    cc_vals = np.zeros(len(v_trials))

    # Mask for the Hα region in observed frame
    search_lo = int(np.searchsorted(wave, ha_rest - 200))
    search_hi = int(np.searchsorted(wave, ha_rest + 200))
    w_search = wave[search_lo:search_hi]
    s_search = central[search_lo:search_hi]

    # Remove continuum baseline with a running median
    baseline = np.zeros_like(s_search)
    halfwin = 50
    for i in range(len(s_search)):
        lo = max(0, i - halfwin)
        hi = min(len(s_search), i + halfwin)
        baseline[i] = np.median(s_search[lo:hi])
    s_cont_sub = s_search - baseline

    # Cross-correlate
    for j, v in enumerate(v_trials):
        z_trial = v / C_KMS
        # Build template on the search wavelength grid
        template = np.zeros(len(w_search))
        for lam_rest, amp in zip(lines, amps):
            lam_obs = lam_rest * (1.0 + z_trial)
            # Place a Gaussian at lam_obs
            template += amp * np.exp(-0.5 * ((w_search - lam_obs) / sigma_aa) ** 2)
        # Normalized cross-correlation
        if np.std(template) > 0 and np.std(s_cont_sub) > 0:
            cc_vals[j] = np.corrcoef(s_cont_sub, template)[0, 1]

    # Best velocity
    best_idx = np.argmax(cc_vals)
    v_sys = v_trials[best_idx]

    # Fine-tune with a Gaussian fit around the best-velocity Hα position
    z_best = v_sys / C_KMS
    ha_obs_guess = ha_rest * (1.0 + z_best)
    fit_lo = int(np.searchsorted(wave, ha_obs_guess - 30))
    fit_hi = int(np.searchsorted(wave, ha_obs_guess + 30))
    if fit_hi - fit_lo >= 10:
        w_fit = wave[fit_lo:fit_hi]
        s_fit = central[fit_lo:fit_hi].astype(np.float64)
        # Remove local baseline
        bl = np.median(s_fit)
        s_fit_sub = s_fit - bl
        # Fit Gaussian
        try:
            from scipy.optimize import curve_fit
            p0 = [np.max(s_fit_sub), ha_obs_guess, 3.0, bl]
            bounds = ([0, ha_obs_guess - 15, 0.5, -np.inf],
                      [np.inf, ha_obs_guess + 15, 20, np.inf])

            def gauss(lam, amp, cen, sig, bl0):
                return amp * np.exp(-0.5 * ((lam - cen) / sig) ** 2) + bl0

            popt, _ = curve_fit(gauss, w_fit, s_fit, p0=p0, bounds=bounds)
            ha_obs = popt[1]
            z_meas = ha_obs / ha_rest - 1.0
            v_sys = z_meas * C_KMS
        except Exception:
            # Fall back to cross-correlation result
            ha_obs = ha_obs_guess
            z_meas = z_best

    # Sanity check: if best correlation is very weak, warn
    if cc_vals[best_idx] < 0.3:
        if verbose:
            print(f"  measure_systemic_velocity: WARNING — weak correlation "
                  f"(r={cc_vals[best_idx]:.3f}). Hα may be weak or absent.")

    if verbose:
        print(f"  measure_systemic_velocity: Hα observed at {ha_obs:.1f} Å "
              f"(rest={ha_rest:.1f}) → z={z_meas:.5f}, V_sys={v_sys:.0f} km/s "
              f"(cc={cc_vals[best_idx]:.3f})")

    hdul.close()
    return v_sys


def load_cube(path=None, memmap=True):
    """Return (data, var, wave, hdr_primary, hdr_data, hdul)."""
    if path is None:
        path = CUBE_PATH
    hdul = fits.open(path, memmap=memmap)
    data = hdul[1].data
    var  = hdul[2].data
    hdr0 = hdul[0].header
    hdr1 = hdul[1].header
    crval3 = hdr1["CRVAL3"]
    crpix3 = hdr1["CRPIX3"]
    cd3    = hdr1["CD3_3"]
    n3     = hdr1["NAXIS3"]
    wave   = crval3 + (np.arange(n3) - (crpix3 - 1)) * cd3
    return data, var, wave, hdr0, hdr1, hdul
