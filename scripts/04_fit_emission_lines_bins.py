"""Stage 4 (bin-level): Per-bin emission line fitting and map construction.

Fits all 12 emission lines simultaneously on each Voronoi bin's
continuum-subtracted spectrum with a shared velocity and velocity dispersion.
[NII]6548/6584 and [OIII]4959/5007 doublet amplitudes are tied at atomic ratios.
MUSE instrumental broadening is accounted for.

Output: bin_kinematics.npz + bin-level intensity maps in bin_line_maps/
"""

import os
import sys
import time
import warnings
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    OUT_DIR, V_SYS, Z_SYS, C_KMS, LINES, DOUBLET_RATIOS, SN_DETECT, muse_lsf_fwhm,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Build the fitting model
# ---------------------------------------------------------------------------
LINE_ORDER = list(LINES.keys())
LINE_WAVES = np.array([LINES[n] for n in LINE_ORDER])
N_LINES = len(LINE_ORDER)

# Free amplitudes: every line except the lower-ratio half of a doublet
FREE_LINES = []
LINKED = {}  # follower -> (leader, ratio = A_follower/A_leader)
for (lead, follow), r in DOUBLET_RATIOS.items():
    LINKED[follow] = (lead, 1.0 / r)  # A(follow) = A(lead) * ratio
for ln in LINE_ORDER:
    if ln not in LINKED:
        FREE_LINES.append(ln)
N_FREE = len(FREE_LINES)
FREE_IDX = {ln: i for i, ln in enumerate(FREE_LINES)}

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def model_function(lam, v_kms, sig_kms, *amps):
    """Sum of Gaussians at rest-frame line wavelengths, shifted by v_kms."""
    z = v_kms / C_KMS
    out = np.zeros_like(lam)
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
        out += A * np.exp(-0.5 * ((lam - lam_obs) / sig_lam) ** 2)
    return out


def integrate_flux_for_lines(v_kms, sig_kms, amps):
    """Integrated fluxes for all lines from fit parameters."""
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
    """Mask of pixels close to any emission line (observed frame)."""
    obs_centers = LINE_WAVES * (1 + Z_SYS)
    keep = np.zeros_like(wave, dtype=bool)
    for c in obs_centers:
        keep |= (wave > c - half_width) & (wave < c + half_width)
    return keep


def main():
    print("Loading bin spectra ...")
    bs = np.load(os.path.join(OUT_DIR, "bin_spectra.npz"))
    bin_spec_sub = bs["bin_spec_sub"]     # (nl, n_bins)
    bin_spec_noise = bs["bin_spec_noise"]
    bin_spec_orig = bs["bin_spec_orig"]
    wave = bs["wave"]
    nl, n_bins = bin_spec_sub.shape
    print(f"  {n_bins} bins, {nl} wavelength pixels")

    # Restrict to channels around the lines
    keep_mask = build_fit_mask(wave, half_width=25.0)
    wave_fit = wave[keep_mask]
    n_chan = keep_mask.sum()
    print(f"  fitting on {n_chan} channels (~{wave_fit[0]:.1f}–{wave_fit[-1]:.1f} Å)")

    # Initial guesses and bounds
    p0 = [V_SYS, 60.0] + [0.0] * N_FREE
    low = [V_SYS - 1500.0, 10.0] + [0.0] * N_FREE
    high = [V_SYS + 1500.0, 400.0] + [np.inf] * N_FREE
    bounds = (low, high)

    # Storage
    v_arr = np.full(n_bins, np.nan, dtype=np.float32)
    sig_arr = np.full(n_bins, np.nan, dtype=np.float32)
    flux_arr = np.full((N_LINES, n_bins), np.nan, dtype=np.float32)
    ferr_arr = np.full((N_LINES, n_bins), np.nan, dtype=np.float32)
    chi2_arr = np.full(n_bins, np.nan, dtype=np.float32)

    t0 = time.time()
    n_fit = 0
    for b in range(n_bins):
        spec = bin_spec_sub[keep_mask, b].astype(np.float64)
        noise = bin_spec_noise[keep_mask, b].astype(np.float64)

        # Check that the bin has usable data
        if not np.all(np.isfinite(spec)) or np.all(noise <= 0):
            continue
        noise = np.clip(noise, 1e-30, None)

        # Amplitude guesses from peak in line cores
        amps_guess = []
        for ln in FREE_LINES:
            lam_obs = LINES[ln] * (1 + Z_SYS)
            ww = (wave_fit > lam_obs - 5) & (wave_fit < lam_obs + 5)
            if ww.sum() > 0:
                pk = np.nanmax(spec[ww])
                amps_guess.append(max(pk, 0.0))
            else:
                amps_guess.append(0.0)
        p0_k = [V_SYS, 60.0] + amps_guess

        try:
            popt, pcov = curve_fit(
                model_function, wave_fit, spec, p0=p0_k,
                sigma=noise, absolute_sigma=False,
                bounds=bounds, maxfev=2000,
            )
            perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
        except Exception:
            continue

        v_kms, sig_kms = popt[0], popt[1]
        amps = popt[2:]
        amp_err = perr[2:]

        fluxes = integrate_flux_for_lines(v_kms, sig_kms, amps)
        z_fit = v_kms / C_KMS
        flux_errs = np.zeros(N_LINES)
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
            flux_errs[i] = dA * np.sqrt(2.0 * np.pi) * sig_lam

        v_arr[b] = v_kms
        sig_arr[b] = sig_kms
        flux_arr[:, b] = fluxes
        ferr_arr[:, b] = flux_errs

        # Chi2
        model = model_function(wave_fit, *popt)
        resid = spec - model
        chi2_arr[b] = np.sum((resid / noise) ** 2) / (n_chan - len(popt))
        n_fit += 1

        if (b + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  bin {b+1}/{n_bins}  v={v_kms:+7.1f}  σ={sig_kms:5.1f}  "
                  f"χ²={chi2_arr[b]:.2f}  elapsed={dt:.1f}s")

    dt = time.time() - t0
    print(f"  {n_fit}/{n_bins} bins fitted in {dt:.1f}s")

    # ---------- Save bin-level results ----------------------------------------
    out_npz = os.path.join(OUT_DIR, "bin_kinematics.npz")
    np.savez(
        out_npz,
        v_kms=v_arr,
        sig_kms=sig_arr,
        fluxes=flux_arr,
        flux_errs=ferr_arr,
        chi2=chi2_arr,
        line_names=np.array(LINE_ORDER, dtype=object),
        free_lines=np.array(FREE_LINES, dtype=object)
    )
    print(f"Wrote {out_npz}")

    # ---------- Build bin-level FITS intensity maps ---------------------------
    # Use the bin map for full field shape
    bin_map = fits.getdata(os.path.join(OUT_DIR, "voronoi_bin_map.fits"))
    ny, nx = bin_map.shape
    vb = np.load(os.path.join(OUT_DIR, "voronoi_bins.npz"))
    bin_num = vb["bin_num"]
    xx = vb["xx"]
    yy = vb["yy"]

    line_dir = os.path.join(OUT_DIR, "bin_line_maps")
    os.makedirs(line_dir, exist_ok=True)

    for i, name in enumerate(LINE_ORDER):
        flux_2d = np.full((ny, nx), np.nan, dtype=np.float32)
        err_2d = np.full((ny, nx), np.nan, dtype=np.float32)
        sn_2d = np.full((ny, nx), np.nan, dtype=np.float32)

        for b in range(n_bins):
            sel = bin_num == b
            if not sel.any():
                continue
            ys_b = yy[sel]
            xs_b = xx[sel]
            if np.isfinite(flux_arr[i, b]):
                flux_2d[ys_b, xs_b] = flux_arr[i, b]
                err_2d[ys_b, xs_b] = ferr_arr[i, b]
                sn = flux_arr[i, b] / ferr_arr[i, b] if ferr_arr[i, b] > 0 else 0.0
                sn_2d[ys_b, xs_b] = sn

        det = sn_2d >= SN_DETECT
        masked_flux = np.where(det, flux_2d, np.nan)

        out_path = os.path.join(line_dir, f"{name}.fits")
        hdul_out = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(flux_2d, name="FLUX"),
            fits.ImageHDU(err_2d, name="FERR"),
            fits.ImageHDU(sn_2d, name="SN"),
            fits.ImageHDU(masked_flux, name="FLUX_SN3"),
        ])
        hdul_out[0].header["LINE"] = name
        hdul_out[0].header["LAM_REST"] = LINES[name]
        hdul_out[0].header["SN_CUT"] = SN_DETECT
        hdul_out.writeto(out_path, overwrite=True)
        n_det = int(np.sum(det))
        print(f"  {name:>9s}  rest={LINES[name]:.2f}  detected bins (S/N≥3) = {n_det}")

    # Also save v/sigma as 2D FITS
    v_map = np.full((ny, nx), np.nan, dtype=np.float32)
    sig_map = np.full((ny, nx), np.nan, dtype=np.float32)
    for b in range(n_bins):
        sel = bin_num == b
        if sel.any():
            v_map[yy[sel], xx[sel]] = v_arr[b]
            sig_map[yy[sel], xx[sel]] = sig_arr[b]

    out_kin = os.path.join(OUT_DIR, "kinematics_bins.fits")
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(v_map, name="V_KMS"),
        fits.ImageHDU(sig_map, name="SIGMA_KMS"),
    ]).writeto(out_kin, overwrite=True)
    print(f"Wrote {out_kin}")
    print("All bin line maps written to", line_dir)


if __name__ == "__main__":
    main()
