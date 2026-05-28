"""Shared constants and helpers for the He 2-10 MUSE pipeline."""
import os
import numpy as np
from astropy.io import fits

ROOT = "/Users/binjia/Desktop/low-metallicity_shocks_LMC"
CUBE_PATH = os.path.join(ROOT, "HEN_2-10.fits")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

C_KMS = 299792.458
V_SYS = 873.0
Z_SYS = V_SYS / C_KMS

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
    # I([OIII]5007) / I([OIII]4959) = 2.98
    ("OIII5007", "OIII4959"): 2.98,
    # I([NII]6584) / I([NII]6548) = 2.94
    ("NII6584", "NII6548"): 2.94,
}

# Continuum reference window for Voronoi S/N (rest-frame Å)
CONT_REST_RANGE = (5300.0, 5530.0)

# Stellar fit wavelength range (rest-frame Å) — MILES coverage minus edges
FIT_REST_RANGE = (4760.0 / (1 + Z_SYS), 7400.0)  # observed left edge in rest

# Local refinement window around emission lines (rest-frame Å half-width)
LOCAL_WIN_HW = 60.0     # 120 Å wide window (Marasco et al. 2023)
LINE_MASK_HW = 12.0     # mask ±12 Å around emission line core during polynomial fit

# pPXF / Voronoi settings
TARGET_SN_BIN = 50.0
SN_FLOOR_VORONOI = 1.0  # skip very low-S/N spaxels

# Detection threshold
SN_DETECT = 3.0


def muse_lsf_fwhm(lam_aa):
    """MUSE LSF FWHM (Å) from Bacon et al. 2017 polynomial fit."""
    return 5.835e-8 * lam_aa ** 2 - 9.080e-4 * lam_aa + 5.983


def load_cube(path=CUBE_PATH, memmap=True):
    """Return (data, var, wave, hdr_primary, hdr_data)."""
    hdul = fits.open(path, memmap=memmap)
    data = hdul[1].data       # (nλ, ny, nx)
    var = hdul[2].data
    hdr0 = hdul[0].header
    hdr1 = hdul[1].header
    crval3 = hdr1["CRVAL3"]
    crpix3 = hdr1["CRPIX3"]
    cd3 = hdr1["CD3_3"]
    n3 = hdr1["NAXIS3"]
    wave = crval3 + (np.arange(n3) - (crpix3 - 1)) * cd3
    return data, var, wave, hdr0, hdr1, hdul
