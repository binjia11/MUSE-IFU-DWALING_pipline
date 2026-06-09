"""Stage 3 (bin-level): Subtract stellar continuum from each Voronoi bin's integrated spectrum.

For each Voronoi bin we:
  1. Sum the spaxel spectra to get the integrated bin spectrum.
  2. Subtract the stage-2 pPXF stellar best-fit directly (no per-spaxel rescaling).
  3. Apply local 3rd-order polynomial refinement around each emission line
     (±60 Å window, masking line core ±12 Å) to correct residual continuum.
  4. Save the continuum-subtracted bin spectra.

Output: bin_spectra.npz  (bin-integrated original, noise, continuum-subtracted, stellar model)
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


def main():
    print("Loading cube ...")
    data, var, wave, hdr0, hdr1, hdul = load_cube()
    nl, ny, nx = data.shape

    vb = np.load(os.path.join(OUT_DIR, "voronoi_bins.npz"))
    bin_num = vb["bin_num"]
    xx = vb["xx"]
    yy = vb["yy"]
    n_bins = int(bin_num.max()) + 1
    print(f"  {n_bins} Voronoi bins, {len(xx)} spaxels")

    fits_in = np.load(os.path.join(OUT_DIR, "ppxf_bin_fits.npz"))
    stellar_models = fits_in["stellar_models"]   # (nl, n_bins)
    i_lo = int(fits_in["i_lo"])
    i_hi = int(fits_in["i_hi"])
    print(f"  stellar models cover px [{i_lo}:{i_hi}] → λ=[{wave[i_lo]:.1f}, {wave[i_hi-1]:.1f}] Å")

    # ---------- Build bin-integrated spectra -----------------------------------
    print("Building bin-integrated spectra ...")
    bin_spec_orig = np.zeros((nl, n_bins), dtype=np.float64)
    bin_spec_noise = np.zeros((nl, n_bins), dtype=np.float64)

    for b in range(n_bins):
        sel = bin_num == b
        ys = yy[sel]
        xs = xx[sel]
        bin_spec_orig[:, b] = data[:, ys, xs].sum(axis=1)
        var_sum = var[:, ys, xs].sum(axis=1)
        bin_spec_noise[:, b] = np.sqrt(np.clip(var_sum, 1e-30, None))

    # ---------- Subtract stellar model -----------------------------------------
    print("Subtracting stellar models ...")
    bin_spec_sub = bin_spec_orig.copy()
    # Stellar model was fit on the bin-integrated spectrum, so subtract directly
    bin_spec_sub[i_lo:i_hi, :] -= stellar_models[i_lo:i_hi, :]

    # ---------- Local polynomial refinement around each line -------------------
    print("Applying local 3rd-order polynomial refinement around each line ...")
    obs_lines = {name: lam * (1 + Z_SYS) for name, lam in LINES.items()}
    t1 = time.time()
    n_refined = 0

    for name, l_obs in obs_lines.items():
        wlo = l_obs - LOCAL_WIN_HW
        whi = l_obs + LOCAL_WIN_HW
        kl = int(np.searchsorted(wave, wlo))
        kh = int(np.searchsorted(wave, whi))
        if kl >= kh - 6:
            continue

        ws = wave[kl:kh]
        n_pix_win = kh - kl

        # Build mask: exclude line core and other lines in this window
        mask = np.ones(n_pix_win, dtype=bool)
        mask &= ~((ws > l_obs - LINE_MASK_HW) & (ws < l_obs + LINE_MASK_HW))
        for name2, l_obs2 in obs_lines.items():
            if name2 == name:
                continue
            mask &= ~((ws > l_obs2 - LINE_MASK_HW) & (ws < l_obs2 + LINE_MASK_HW))
        if mask.sum() < 8:
            continue

        x_centered = (ws - l_obs).astype(np.float64)
        X_full = np.vander(x_centered, 4, increasing=True)  # (n_pix, 4)
        X_masked = X_full[mask]                             # (n_good, 4)

        for b in range(n_bins):
            yvals = bin_spec_sub[kl:kh, b].astype(np.float64)
            if not np.all(np.isfinite(yvals)):
                continue
            coeffs, *_ = np.linalg.lstsq(X_masked, yvals[mask], rcond=None)
            baseline = X_full @ coeffs
            bin_spec_sub[kl:kh, b] -= baseline
        n_refined += 1
        print(f"  {name}: window=[{wlo:.1f},{whi:.1f}] mask_keep={mask.sum()}")

    print(f"  Local refinement applied to {n_refined}/{len(LINES)} lines in {time.time()-t1:.1f}s")

    # ---------- Save -----------------------------------------------------------
    out_path = os.path.join(OUT_DIR, "bin_spectra.npz")
    np.savez(
        out_path,
        wave=wave,
        bin_spec_orig=bin_spec_orig.astype(np.float32),
        bin_spec_noise=bin_spec_noise.astype(np.float32),
        bin_spec_sub=bin_spec_sub.astype(np.float32),
        stellar_models=stellar_models,
        i_lo=i_lo, i_hi=i_hi,
    )
    print(f"Wrote {out_path}")
    hdul.close()


if __name__ == "__main__":
    main()