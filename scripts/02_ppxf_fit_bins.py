"""Stage 2: Fit SSP templates with pPXF on each Voronoi bin.

For each bin we
  - sum the spaxel spectra & propagate variance,
  - log-rebin galaxy + templates to a common velscale,
  - fit pPXF with stars + Gaussian gas templates,
  - store the stellar best-fit (without gas), gas best-fit, and parameters.

Supports BPASS v2.2.1 (binary, 1 Myr–100 Gyr) and EMILES (single-star, 30 Myr–14 Gyr).
Set TEMPLATE_LIB below to switch.
"""
import os
import sys
import time
import warnings
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    CUBE_PATH, OUT_DIR, V_SYS, Z_SYS, C_KMS, load_cube, muse_lsf_fwhm, ROOT,
)

import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from ppxf.ppxf import ppxf

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TEMPLATE_LIB = os.environ.get("PIPE_TEMPLATE_LIB", "bpass")  # "bpass" or "emiles"

SPS_NPZ = os.path.join(
    os.path.dirname(__import__("ppxf").__file__),
    "sps_models", "spectra_emiles_9.0.npz",
)
BPASS_NPZ = os.path.join(ROOT, "templates", "bpass_processed", "bpass_templates_raw.npz")

VELSCALE_OUT = 60.0          # km/s / pixel in log-rebinned spectra
FIT_OBS_RANGE = (4760.0 * (1 + Z_SYS), 7400.0 * (1 + Z_SYS))   # observed-frame fit window
# Telluric sky-line masks (observed-frame wavelengths), converted to rest below
_SKY_MASK_OBS = [(5570.0, 5590.0), (6290.0, 6310.0), (6360.0, 6370.0)]
SKY_MASK_REST = [(lo / (1 + Z_SYS), hi / (1 + Z_SYS)) for lo, hi in _SKY_MASK_OBS]

REGUL_ERR = 0.013            # pPXF regul ~ 1/std(noise)


def load_bpass_templates(ln_lam_gal, velscale, FWHM_gal):
    """Load BPASS v2.2.1 templates, log-rebin, convolve to MUSE LSF.

    Returns (templates_2d, ln_lam_out) where templates_2d has shape
    (n_pix_log, n_ages * n_metals).  The output wavelength grid is
    extended by ~4% on each side so pPXF's internal velocity-shift
    padding is covered.
    """
    data = np.load(BPASS_NPZ, allow_pickle=True)
    wave_bpass = data["wave"]
    z_strings = data["z_strings"]
    ages_gyr = data["ages_gyr"]
    templates_dict = data["templates_dict"].item()

    lam_gal = np.exp(ln_lam_gal)
    d_ln_lam = ln_lam_gal[1] - ln_lam_gal[0]

    # Extend the template log-grid by ±4% in wavelength
    pad_pix = int(0.04 / d_ln_lam)  # ~4% padding
    ln_lam_temp = np.linspace(
        ln_lam_gal[0] - pad_pix * d_ln_lam,
        ln_lam_gal[-1] + pad_pix * d_ln_lam,
        len(ln_lam_gal) + 2 * pad_pix,
    )
    lam_temp_arr = np.exp(ln_lam_temp)

    all_flux_log = []
    for zstr in z_strings:
        fluxes = templates_dict[zstr]  # (n_wave_bpass, n_ages)
        n_ages = fluxes.shape[1]

        for a in range(n_ages):
            # Interpolate from linear BPASS grid to the extended log grid
            flux_lin = np.interp(lam_temp_arr, wave_bpass, fluxes[:, a])

            # Convolve to MUSE LSF
            if isinstance(FWHM_gal, dict):
                fwhm_interp = np.interp(lam_temp_arr, FWHM_gal["lam"], FWHM_gal["fwhm"])
            else:
                fwhm_interp = np.full_like(lam_temp_arr, FWHM_gal)
            sig_pix = (fwhm_interp / 2.355) / (lam_temp_arr * velscale / C_KMS)
            med_sig = np.median(sig_pix)

            if med_sig > 0.5:
                flux_conv = gaussian_filter1d(flux_lin, med_sig, mode='nearest')
            else:
                flux_conv = flux_lin

            all_flux_log.append(flux_conv)

    templates = np.column_stack(all_flux_log)  # (n_pix_log_extended, n_ages * n_metals)
    templates /= np.median(templates)
    print(f"  BPASS: {templates.shape[1]} templates ({len(z_strings)} Z × {n_ages} ages)"
          f"  ages={ages_gyr[0]*1000:.0f} Myr–{ages_gyr[-1]:.0f} Gyr"
          f"  Z={list(z_strings)}"
          f"  λ=[{lam_temp_arr[0]:.1f}, {lam_temp_arr[-1]:.1f}] Å")
    return templates, ln_lam_temp


def integrated_spectrum(data, var, bin_num, xx, yy, b):
    """Sum spaxels belonging to bin b → (spec, noise) (flux density preserving)."""
    sel = bin_num == b
    ys = yy[sel]
    xs = xx[sel]
    spec = data[:, ys, xs].sum(axis=1)
    var_sum = var[:, ys, xs].sum(axis=1)
    noise = np.sqrt(np.clip(var_sum, 1e-30, None))
    return spec.astype(np.float64), noise.astype(np.float64), sel.sum()


def main():
    print("Loading cube ...")
    data, var, wave, hdr0, hdr1, hdul = load_cube()
    nl, ny, nx = data.shape

    vb = np.load(os.path.join(OUT_DIR, "voronoi_bins.npz"))
    bin_num = vb["bin_num"]
    xx = vb["xx"]
    yy = vb["yy"]
    n_bins = int(bin_num.max()) + 1
    print(f"  {n_bins} Voronoi bins")

    # ---------- prepare fit window in observed frame ------------------------
    i_lo = int(np.searchsorted(wave, FIT_OBS_RANGE[0]))
    i_hi = int(np.searchsorted(wave, FIT_OBS_RANGE[1]))
    wave_fit = wave[i_lo:i_hi]
    print(f"Fit window (obs) = [{wave_fit[0]:.2f}, {wave_fit[-1]:.2f}] Å  ({len(wave_fit)} px)")

    # MUSE LSF as a wavelength-dependent FWHM dict (Bacon+ 2017)
    fwhm_muse = muse_lsf_fwhm(wave_fit)
    FWHM_gal = {"lam": wave_fit, "fwhm": fwhm_muse}

    # ---------- Log-rebin a sample bin to get common velscale --------------
    print("Probing log-rebin velscale ...")
    sample_b = int(np.argmax(np.bincount(bin_num)))
    spec0, noise0, npix0 = integrated_spectrum(data, var, bin_num, xx, yy, sample_b)
    spec_fit = spec0[i_lo:i_hi]
    lam_range_gal = np.array([wave_fit[0], wave_fit[-1]]) / (1 + Z_SYS)

    galaxy_log, ln_lam_gal, velscale = util.log_rebin(lam_range_gal, spec_fit)
    print(f"  velscale = {velscale:.3f} km/s/px")

    # ---------- Load stellar templates ---------------------------------------
    if TEMPLATE_LIB == "bpass" and os.path.exists(BPASS_NPZ):
        print(f"Loading BPASS v2.2.1 from {BPASS_NPZ} ...")
        templates, ln_lam_temp = load_bpass_templates(ln_lam_gal, velscale, FWHM_gal)
        n_temps_stars = templates.shape[1]
    else:
        print(f"Loading EMILES from {SPS_NPZ} ...")
        sps = lib.sps_lib(
            SPS_NPZ, velscale=velscale, fwhm_gal=FWHM_gal,
            norm_range=[5070, 5950], lam_range=[3540, 7500],
        )
        templates = sps.templates.reshape(sps.templates.shape[0], -1)
        templates /= np.median(templates)
        ln_lam_temp = sps.ln_lam_temp
        n_temps_stars = templates.shape[1]
    print(f"  {n_temps_stars} stellar templates, npix={templates.shape[0]}")

    # ---------- Gas templates for joint fit ---------------------------------
    gas_templates, gas_names, gas_wave = util.emission_lines(
        ln_lam_temp, lam_range_gal, FWHM_gal,
        tie_balmer=False, limit_doublets=False,
    )
    # Sort: Balmer first, then forbidden
    balmer_mask = np.array(["[" not in n for n in gas_names])
    order_gas = np.concatenate([np.where(balmer_mask)[0], np.where(~balmer_mask)[0]])
    gas_templates = gas_templates[:, order_gas]
    gas_names = gas_names[order_gas]
    gas_wave = gas_wave[order_gas]
    n_balmer = int(balmer_mask.sum())
    n_forbidden = int((~balmer_mask).sum())
    print(f"  {n_balmer+n_forbidden} gas templates  Balmer={n_balmer}, forbidden={n_forbidden}")
    print(f"  gas names: {list(gas_names)}")

    component = (
        [0] * n_temps_stars
        + [1] * n_balmer
        + [2] * n_forbidden
    )
    gas_component = np.array(component) > 0
    moments = [2, 2, 2]                    # V, σ per component

    V0 = 0.0        # both templates and galaxy are in rest frame
    start = [
        [V0, 60.0],                       # stars
        [V0, 60.0],                       # Balmer
        [V0, 60.0],                       # forbidden
    ]

    all_templates = np.column_stack([templates, gas_templates])

    # Storage arrays
    stellar_models = np.zeros((nl, n_bins), dtype=np.float32)   # linear-λ stellar continuum
    gas_models = np.zeros((nl, n_bins), dtype=np.float32)       # linear-λ gas model (for diagnostic)
    par_records = []

    t0 = time.time()
    for b in range(n_bins):
        spec, noise, npx = integrated_spectrum(data, var, bin_num, xx, yy, b)
        spec_fit = spec[i_lo:i_hi]
        noise_fit = noise[i_lo:i_hi]
        # Replace NaNs/zeros for robustness
        bad = ~np.isfinite(spec_fit) | ~np.isfinite(noise_fit) | (noise_fit <= 0)
        if bad.any():
            spec_fit = spec_fit.copy()
            noise_fit = noise_fit.copy()
            spec_fit[bad] = np.nanmedian(spec_fit)
            noise_fit[bad] = np.nanmedian(noise_fit) * 10

        galaxy_log, ln_lam_gal, _ = util.log_rebin(lam_range_gal, spec_fit, velscale=velscale)
        noise_log, _, _ = util.log_rebin(lam_range_gal, noise_fit, velscale=velscale)
        # Normalise for numerical stability
        norm = np.median(galaxy_log)
        if not np.isfinite(norm) or norm <= 0:
            norm = 1.0
        galaxy_log = galaxy_log / norm
        noise_log = np.clip(noise_log / norm, 1e-5, None)

        # Build sky mask in log-space
        lam_gal = np.exp(ln_lam_gal)
        good_pix = np.ones_like(lam_gal, dtype=bool)
        for lo, hi in SKY_MASK_REST:
            good_pix &= ~((lam_gal > lo) & (lam_gal < hi))
        good_pixels = np.where(good_pix)[0]

        # Initial velocity guess (rest -> observed shift relative to lam_range_gal[0])
        # Already in observed frame so V=0 is fine; gas allowed to drift.
        try:
            pp = ppxf(
                all_templates, galaxy_log, noise_log, velscale,
                start, plot=False, moments=moments,
                degree=-1, mdegree=10,
                component=component, gas_component=gas_component,
                gas_names=gas_names, lam=lam_gal, lam_temp=np.exp(ln_lam_temp),
                goodpixels=good_pixels, quiet=True,
            )
        except Exception as e:
            print(f"  bin {b}: pPXF failed ({e})")
            continue

        # Stellar best-fit (no gas) in log-rebinned grid → rescale back & interp to linear
        bestfit_log = pp.bestfit
        gas_bestfit_log = pp.gas_bestfit if pp.gas_bestfit is not None else np.zeros_like(bestfit_log)
        star_log = bestfit_log - gas_bestfit_log
        # Unnormalise
        star_log *= norm
        gas_bestfit_log = gas_bestfit_log * norm

        # Interpolate back to linear wave grid (only inside fit window)
        star_lin = np.interp(wave_fit, lam_gal, star_log)
        gas_lin = np.interp(wave_fit, lam_gal, gas_bestfit_log)

        stellar_models[i_lo:i_hi, b] = star_lin
        gas_models[i_lo:i_hi, b] = gas_lin

        # Record parameters
        sol = pp.sol if isinstance(pp.sol, list) else [pp.sol]
        v_star, sig_star = sol[0][0], sol[0][1]
        v_balmer, sig_balmer = (sol[1][0], sol[1][1]) if len(sol) > 1 else (np.nan, np.nan)
        v_forbid, sig_forbid = (sol[2][0], sol[2][1]) if len(sol) > 2 else (np.nan, np.nan)
        chi2 = pp.chi2
        par_records.append((b, npx, v_star, sig_star, v_balmer, sig_balmer, v_forbid, sig_forbid, chi2))

        if (b + 1) % 10 == 0 or b == n_bins - 1:
            dt = time.time() - t0
            eta = dt / (b + 1) * (n_bins - b - 1)
            print(f"  bin {b+1}/{n_bins}  npix={npx:5d}  v*={v_star:+7.1f}  σ*={sig_star:5.1f}  "
                  f"χ2={chi2:.2f}  elapsed={dt:5.1f}s  eta={eta:5.1f}s")

    # Save outputs
    out_npz = os.path.join(OUT_DIR, "ppxf_bin_fits.npz")
    np.savez(
        out_npz,
        wave=wave, i_lo=i_lo, i_hi=i_hi,
        stellar_models=stellar_models,
        gas_models=gas_models,
        params=np.array(par_records, dtype=float),
        param_cols=np.array(
            ["bin", "npix", "v_star", "sigma_star",
             "v_balmer", "sigma_balmer", "v_forbid", "sigma_forbid", "chi2"],
            dtype=object,
        ),
    )
    print(f"Wrote {out_npz}")
    hdul.close()


if __name__ == "__main__":
    main()
