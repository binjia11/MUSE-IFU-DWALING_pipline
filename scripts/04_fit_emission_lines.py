"""Stage 4: Per-spaxel emission line fitting and map construction.

Following Cresci+2017 §2.1 we fit *all* lines simultaneously in each spaxel
with a single shared velocity and velocity dispersion (Gaussian profile),
leaving amplitudes free except for the [NII]6548/6584 and [OIII]4959/5007
doublets, which are tied at their atomic ratios.  We also include the MUSE
instrumental broadening when converting between the Gaussian σ in
wavelength and the intrinsic gas σ in km/s.

For each line we compute
    F_line = A_λ · √(2π) · σ_λ_obs               (integrated flux in cube units)
and propagate the formal covariance from `scipy.optimize.curve_fit` into
F_err.  A spaxel pixel is marked DETECTED when F_line / F_err ≥ 3.
"""
import os
import sys
import time
import warnings
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    OUT_DIR, GALAXY, V_SYS, Z_SYS, C_KMS, LINES, DOUBLET_RATIOS, SN_DETECT,
    muse_lsf_fwhm, load_cube,
)

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Build the fitting model — one shared v, σ; amplitudes tied for doublets
# -----------------------------------------------------------------------------
LINE_ORDER = list(LINES.keys())              # canonical order
LINE_WAVES = np.array([LINES[n] for n in LINE_ORDER])
N_LINES = len(LINE_ORDER)

# free amplitudes: every line except the lower-ratio half of a doublet
FREE_LINES = []
LINKED = {}                                  # follower -> (leader, ratio_follower/leader)
for (lead, follow), r in DOUBLET_RATIOS.items():
    LINKED[follow] = (lead, 1.0 / r)         # A(follow) = A(lead)/r
for ln in LINE_ORDER:
    if ln not in LINKED:
        FREE_LINES.append(ln)
N_FREE = len(FREE_LINES)
FREE_IDX = {ln: i for i, ln in enumerate(FREE_LINES)}

# -----------------------------------------------------------------------------
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def model_function(lam, v_kms, sig_kms, *amps):
    """Sum of Gaussians at the rest-frame line wavelengths, shifted by v_kms."""
    # observed wavelength of each line (galaxy + peculiar v)
    z = v_kms / C_KMS
    out = np.zeros_like(lam)
    # instrumental sigma in wavelength at the observed wavelength of each line
    for i, (name, lam0) in enumerate(zip(LINE_ORDER, LINE_WAVES)):
        lam_obs = lam0 * (1.0 + z)
        if name in LINKED:
            leader, ratio = LINKED[name]
            A = amps[FREE_IDX[leader]] * ratio
        else:
            A = amps[FREE_IDX[name]]
        # Gas σ in wavelength
        sig_lam_gas = (sig_kms / C_KMS) * lam_obs
        # Instrumental σ
        fwhm_inst = muse_lsf_fwhm(lam_obs)
        sig_lam_inst = fwhm_inst * FWHM_TO_SIGMA
        sig_lam = np.sqrt(sig_lam_gas ** 2 + sig_lam_inst ** 2)
        out += A * np.exp(-0.5 * ((lam - lam_obs) / sig_lam) ** 2)
    return out


def integrate_flux_for_lines(v_kms, sig_kms, amps):
    """Return integrated fluxes for *all* LINE_ORDER lines from the fit params."""
    z = v_kms / C_KMS
    fluxes = np.zeros(N_LINES)
    for i, (name, lam0) in enumerate(zip(LINE_ORDER, LINE_WAVES)):
        lam_obs = lam0 * (1.0 + z)
        if name in LINKED:
            leader, ratio = LINKED[name]
            A = amps[FREE_IDX[leader]] * ratio
        else:
            A = amps[FREE_IDX[name]]
        sig_lam_gas = (sig_kms / C_KMS) * lam_obs
        fwhm_inst = muse_lsf_fwhm(lam_obs)
        sig_lam_inst = fwhm_inst * FWHM_TO_SIGMA
        sig_lam = np.sqrt(sig_lam_gas ** 2 + sig_lam_inst ** 2)
        fluxes[i] = A * np.sqrt(2.0 * np.pi) * sig_lam
    return fluxes


def build_fit_mask(wave, half_width=25.0):
    """Mask in wavelength of pixels close to *any* line (used by fitter)."""
    z = Z_SYS
    obs_centers = LINE_WAVES * (1 + z)
    keep = np.zeros_like(wave, dtype=bool)
    for c in obs_centers:
        keep |= (wave > c - half_width) & (wave < c + half_width)
    return keep


# -----------------------------------------------------------------------------
def fit_one_spaxel(args):
    iy, ix, spec, noise, wave_fit, p0, bounds = args
    try:
        popt, pcov = curve_fit(
            model_function, wave_fit, spec, p0=p0,
            sigma=noise, absolute_sigma=False,
            bounds=bounds, maxfev=2000,
        )
        perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    except Exception:
        return iy, ix, None, None
    v_kms, sig_kms = popt[0], popt[1]
    amps = popt[2:]
    amp_err = perr[2:]
    # Integrated fluxes & errors per line (propagated from amplitude err only,
    # which dominates over v/σ uncertainty for detection-level S/N)
    fluxes = integrate_flux_for_lines(v_kms, sig_kms, amps)
    # σ_λ_obs at each line — same prefactor used for error propagation
    z_fit = v_kms / C_KMS
    flux_err = np.zeros(N_LINES)
    for i, (name, lam0) in enumerate(zip(LINE_ORDER, LINE_WAVES)):
        lam_obs = lam0 * (1.0 + z_fit)
        sig_lam_gas = (sig_kms / C_KMS) * lam_obs
        fwhm_inst = muse_lsf_fwhm(lam_obs)
        sig_lam_inst = fwhm_inst * FWHM_TO_SIGMA
        sig_lam = np.sqrt(sig_lam_gas ** 2 + sig_lam_inst ** 2)
        if name in LINKED:
            leader, ratio = LINKED[name]
            dA = amp_err[FREE_IDX[leader]] * ratio
        else:
            dA = amp_err[FREE_IDX[name]]
        flux_err[i] = dA * np.sqrt(2.0 * np.pi) * sig_lam
    return iy, ix, (v_kms, sig_kms, fluxes), flux_err


def main():
    print("Loading subtracted cube ...")
    hdul = fits.open(os.path.join(OUT_DIR, f"{GALAXY}_cont_sub.fits"), memmap=True)
    data = hdul[1].data        # already continuum-subtracted
    var = hdul[2].data
    valid = hdul[3].data.astype(bool)
    hdr = hdul[1].header
    n3 = hdr["NAXIS3"]
    crval3 = hdr["CRVAL3"]; crpix3 = hdr["CRPIX3"]; cd3 = hdr["CD3_3"]
    wave = crval3 + (np.arange(n3) - (crpix3 - 1)) * cd3
    print(f"  shape={data.shape}  valid spaxels={int(valid.sum())}")

    # Restrict to wavelength channels around the lines
    keep_mask = build_fit_mask(wave, half_width=25.0)
    wave_fit = wave[keep_mask]
    print(f"  fitting on {keep_mask.sum()} channels (∼{wave_fit[0]:.1f}–{wave_fit[-1]:.1f} Å)")

    yy, xx = np.where(valid)
    n_spx = len(yy)
    print(f"  {n_spx} spaxels to fit")

    # Initial guesses & bounds
    p0 = [V_SYS, 60.0] + [0.0] * N_FREE
    # Bound velocity ±1500 km/s, σ in [10, 400] km/s, amplitudes ≥ 0
    low = [V_SYS - 1500.0, 10.0] + [0.0] * N_FREE
    high = [V_SYS + 1500.0, 400.0] + [np.inf] * N_FREE
    bounds = (low, high)

    # Build a per-spaxel iterator that avoids passing the huge cube each time
    def gen():
        for k in range(n_spx):
            sy, sx = yy[k], xx[k]
            spec = data[keep_mask, sy, sx].astype(np.float64)
            n2 = var[keep_mask, sy, sx].astype(np.float64)
            n2 = np.clip(n2, 1e-30, None)
            noise = np.sqrt(n2)
            # mild guess on amplitudes: max flux in line core
            amps_guess = []
            z = Z_SYS
            for ln in FREE_LINES:
                lam0 = LINES[ln] * (1 + z)
                ww = (wave_fit > lam0 - 5) & (wave_fit < lam0 + 5)
                if ww.sum() > 0:
                    amps_guess.append(max(np.nanmax(spec[ww]), 0))
                else:
                    amps_guess.append(0.0)
            p0_k = [V_SYS, 60.0] + amps_guess
            yield (sy, sx, spec, noise, wave_fit, p0_k, bounds)

    t0 = time.time()
    flux_cube = np.zeros((N_LINES, data.shape[1], data.shape[2]), dtype=np.float32)
    err_cube = np.zeros_like(flux_cube)
    v_map = np.full((data.shape[1], data.shape[2]), np.nan, dtype=np.float32)
    sig_map = np.full_like(v_map, np.nan)

    n_proc = max(1, (os.cpu_count() or 4) - 1)
    print(f"  using {n_proc} worker processes")

    with Pool(n_proc) as pool:
        done = 0
        for iy, ix, params, ferr in pool.imap_unordered(fit_one_spaxel, gen(), chunksize=64):
            done += 1
            if params is None:
                continue
            v, s, f = params
            v_map[iy, ix] = v
            sig_map[iy, ix] = s
            flux_cube[:, iy, ix] = f
            err_cube[:, iy, ix] = ferr
            if done % 1000 == 0:
                dt = time.time() - t0
                eta = dt / done * (n_spx - done)
                print(f"  {done}/{n_spx}  elapsed={dt:.0f}s  eta={eta:.0f}s")

    print(f"Done all spaxels in {time.time()-t0:.1f}s")

    # Save kinematic maps + per-line flux/err/SN cubes
    out_kin = os.path.join(OUT_DIR, "kinematics.fits")
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(v_map, header=hdr, name="V_KMS"),
        fits.ImageHDU(sig_map, header=hdr, name="SIGMA_KMS"),
    ]).writeto(out_kin, overwrite=True)
    print(f"Wrote {out_kin}")

    line_dir = os.path.join(OUT_DIR, "line_maps")
    os.makedirs(line_dir, exist_ok=True)
    for i, name in enumerate(LINE_ORDER):
        flux = flux_cube[i]
        err = err_cube[i]
        sn = np.where(err > 0, flux / err, 0.0)
        det = sn >= SN_DETECT
        masked_flux = np.where(det, flux, np.nan)
        out_path = os.path.join(line_dir, f"{name}.fits")
        hdul_out = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(flux.astype(np.float32), header=hdr, name="FLUX"),
            fits.ImageHDU(err.astype(np.float32), header=hdr, name="FERR"),
            fits.ImageHDU(sn.astype(np.float32), header=hdr, name="SN"),
            fits.ImageHDU(masked_flux.astype(np.float32), header=hdr, name="FLUX_SN3"),
        ])
        hdul_out[0].header["LINE"] = name
        hdul_out[0].header["LAM_REST"] = LINES[name]
        hdul_out[0].header["SN_CUT"] = SN_DETECT
        hdul_out.writeto(out_path, overwrite=True)
        n_det = int(np.sum(det))
        print(f"  {name:>9s}  rest={LINES[name]:.2f}  detected pixels (S/N≥3) = {n_det}")
    hdul.close()
    print("All line maps written to", line_dir)


if __name__ == "__main__":
    main()
