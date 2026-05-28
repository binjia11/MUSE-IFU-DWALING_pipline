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
FIT_REST_RANGE = (4760.0 / (1 + Z_SYS), 7400.0)

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
