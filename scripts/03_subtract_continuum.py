"""Stage 3: Subtract the rescaled stellar model from each spaxel.

For each Voronoi bin, the stage-2 pPXF run produced a stellar best-fit on
the linear MUSE wavelength grid (over the MILES coverage window).  We

  1. Rescale this model per-spaxel to match the local median continuum.
  2. Subtract it from the spaxel spectrum.
  3. Outside the MILES coverage (notably around [SIII]9069), we apply a
     local 3rd-order polynomial fit (Marasco et al. 2023) inside ±60 Å of
     each emission line, masking the line core (±12 Å).
  4. Save the continuum-subtracted cube (DATA + STAT extensions preserved).
"""
import os
import sys
import time
import warnings
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    CUBE_PATH, OUT_DIR, GALAXY, Z_SYS, LINES, LOCAL_WIN_HW, LINE_MASK_HW, load_cube,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


CONT_MED_RANGE_OBS = (5300.0 * (1 + Z_SYS), 5530.0 * (1 + Z_SYS))


def main():
    print("Loading cube ...")
    data, var, wave, hdr0, hdr1, hdul = load_cube()
    nl, ny, nx = data.shape

    vb = np.load(os.path.join(OUT_DIR, "voronoi_bins.npz"))
    bin_num = vb["bin_num"]
    xx = vb["xx"]
    yy = vb["yy"]
    n_bins = int(bin_num.max()) + 1

    fits_in = np.load(os.path.join(OUT_DIR, "ppxf_bin_fits.npz"))
    stellar_models = fits_in["stellar_models"]   # (nl, n_bins)
    i_lo = int(fits_in["i_lo"])
    i_hi = int(fits_in["i_hi"])
    print(f"  Stellar models cover px [{i_lo}:{i_hi}] → λ=[{wave[i_lo]:.1f}, {wave[i_hi-1]:.1f}] Å")

    # Window used to compute the per-spaxel rescaling factor
    j_lo = int(np.searchsorted(wave, CONT_MED_RANGE_OBS[0]))
    j_hi = int(np.searchsorted(wave, CONT_MED_RANGE_OBS[1]))

    # Median of each bin's stellar model in the same window (denominator)
    bin_model_med = np.nanmedian(stellar_models[j_lo:j_hi, :], axis=0)
    bin_model_med = np.where(bin_model_med > 0, bin_model_med, np.nan)

    # ---------- Step 1+2: Subtract rescaled bin model ----------------------
    print("Subtracting rescaled stellar model from each spaxel ...")
    cube_sub = np.array(data, dtype=np.float32, copy=True)  # full sub cube
    # cube_sub remains == data outside galaxy mask (we'll zero later via header mask)

    valid_pix_map = np.zeros((ny, nx), dtype=bool)
    t0 = time.time()
    chunk = 500
    for ck in range(0, len(xx), chunk):
        ie = min(ck + chunk, len(xx))
        xs = xx[ck:ie]
        ys = yy[ck:ie]
        bs = bin_num[ck:ie]
        # local median per spaxel
        slabs = data[j_lo:j_hi, ys, xs]            # (npix_window, n_spx)
        local_med = np.nanmedian(slabs, axis=0)    # (n_spx,)
        # rescale factors
        bin_med = bin_model_med[bs]
        scale = local_med / bin_med
        scale = np.where(np.isfinite(scale) & (bin_med > 0), scale, 0.0)
        # broadcast subtraction
        for k, (sx, sy, sb, sc) in enumerate(zip(xs, ys, bs, scale)):
            if not np.isfinite(sc):
                continue
            cube_sub[i_lo:i_hi, sy, sx] -= sc * stellar_models[i_lo:i_hi, sb]
            valid_pix_map[sy, sx] = True
        if (ie // chunk) % 5 == 0:
            print(f"  {ie}/{len(xx)} spaxels done  ({time.time()-t0:.1f}s)")
    print(f"  Step 1 done in {time.time()-t0:.1f}s; valid spaxels = {valid_pix_map.sum()}")

    # ---------- Step 3: Local polynomial refinement around each line -------
    print("Applying local 3rd-order polynomial refinement around each line ...")
    # We refine in OBSERVED frame: window = line_obs ± LOCAL_WIN_HW Å
    obs_lines = {name: lam * (1 + Z_SYS) for name, lam in LINES.items()}
    yy_v = yy
    xx_v = xx
    n_spx = len(xx_v)

    # Restrict refinement to spaxels that pass valid_pix_map
    t1 = time.time()
    for name, l_obs in obs_lines.items():
        wlo = l_obs - LOCAL_WIN_HW
        whi = l_obs + LOCAL_WIN_HW
        kl = int(np.searchsorted(wave, wlo))
        kh = int(np.searchsorted(wave, whi))
        if kl >= kh - 4:
            continue
        ws = wave[kl:kh]
        # build per-line mask of OTHER lines + central core
        mask = np.ones(kh - kl, dtype=bool)
        # mask the central line ±LINE_MASK_HW
        mask &= ~((ws > l_obs - LINE_MASK_HW) & (ws < l_obs + LINE_MASK_HW))
        # mask any other lines that fall inside the window
        for name2, l_obs2 in obs_lines.items():
            if name2 == name:
                continue
            mask &= ~((ws > l_obs2 - LINE_MASK_HW) & (ws < l_obs2 + LINE_MASK_HW))
        if mask.sum() < 6:
            continue

        # Vectorise over all spaxels at once
        slab = cube_sub[kl:kh, yy_v, xx_v]         # (npix_win, n_spx)
        good_pix = mask[:, None] & np.isfinite(slab)
        # For points with too few good_pix per spaxel, skip refinement
        n_good = good_pix.sum(axis=0)
        # Fit polynomial of order 3 = 4 coefficients, need at least 6 good points
        do_fit = n_good >= 8
        if not np.any(do_fit):
            continue

        # Build design matrix once
        x_centered = (ws - l_obs).astype(np.float32)
        X_full = np.vander(x_centered, 4, increasing=True).astype(np.float32)  # (npix, 4)

        # Loop in a vectorised batch — use lstsq per-spaxel
        idxs = np.where(do_fit)[0]
        for j in idxs:
            yvals = slab[:, j].astype(np.float64)
            m = good_pix[:, j]
            if m.sum() < 8:
                continue
            coeffs, *_ = np.linalg.lstsq(X_full[m].astype(np.float64), yvals[m], rcond=None)
            baseline = X_full.astype(np.float64) @ coeffs
            cube_sub[kl:kh, yy_v[j], xx_v[j]] -= baseline.astype(np.float32)
        print(f"  {name}: window=[{wlo:.1f},{whi:.1f}] mask_keep={mask.sum()} fits={do_fit.sum()}")

    print(f"  Step 3 done in {time.time()-t1:.1f}s")

    # ---------- Save subtracted cube ---------------------------------------
    out_path = os.path.join(OUT_DIR, f"{GALAXY}_cont_sub.fits")
    print(f"Writing {out_path} ...")
    hdu0 = fits.PrimaryHDU(header=hdr0)
    hdu1 = fits.ImageHDU(cube_sub.astype(np.float32), header=hdr1, name="DATA")
    hdu2 = fits.ImageHDU(var.astype(np.float32), header=hdul[2].header, name="STAT")
    hdu3 = fits.ImageHDU(valid_pix_map.astype(np.uint8), header=hdr1, name="VALID")
    fits.HDUList([hdu0, hdu1, hdu2, hdu3]).writeto(out_path, overwrite=True)
    print("Done.")
    hdul.close()


if __name__ == "__main__":
    main()
