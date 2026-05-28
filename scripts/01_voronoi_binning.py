"""Stage 1: Compute continuum S/N per spaxel and Voronoi-bin to S/N=50.

Follows Cresci et al. 2017 §2.1: continuum S/N is measured in 5300–5530 Å
(rest-frame, below the stellar fit upper bound), and Voronoi binning is
applied to achieve a minimum S/N of 50 per bin on the continuum.
"""
import os
import sys
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    CUBE_PATH, OUT_DIR, Z_SYS, CONT_REST_RANGE, TARGET_SN_BIN,
    SN_FLOOR_VORONOI, load_cube,
)
from vorbin.voronoi_2d_binning import voronoi_2d_binning


def main():
    print("Loading cube ...")
    data, var, wave, hdr0, hdr1, hdul = load_cube()
    nl, ny, nx = data.shape
    print(f"  cube shape = ({nl}, {ny}, {nx})")
    print(f"  wave = [{wave[0]:.2f}, {wave[-1]:.2f}] Å")

    # Continuum reference window (rest-frame -> observed)
    l_lo = CONT_REST_RANGE[0] * (1 + Z_SYS)
    l_hi = CONT_REST_RANGE[1] * (1 + Z_SYS)
    i_lo = int(np.searchsorted(wave, l_lo))
    i_hi = int(np.searchsorted(wave, l_hi))
    print(f"Continuum window (obs): [{wave[i_lo]:.2f}, {wave[i_hi]:.2f}] Å  ({i_hi-i_lo} px)")

    # Compute signal (median) and noise (median sqrt(var)) per spaxel
    print("Computing per-spaxel continuum signal and noise ...")
    slab = data[i_lo:i_hi, :, :].astype(np.float32)
    var_slab = var[i_lo:i_hi, :, :].astype(np.float32)
    # Mask NaN / non-finite
    bad = ~np.isfinite(slab) | ~np.isfinite(var_slab) | (var_slab <= 0)
    slab[bad] = np.nan
    var_slab[bad] = np.nan
    signal = np.nanmedian(slab, axis=0)                       # (ny, nx)
    noise = np.sqrt(np.nanmedian(var_slab, axis=0))           # (ny, nx)
    sn = signal / noise

    # Save quicklook S/N map
    fits.PrimaryHDU(sn.astype(np.float32), header=hdr1).writeto(
        os.path.join(OUT_DIR, "cont_SN_map.fits"), overwrite=True
    )
    fits.PrimaryHDU(signal.astype(np.float32), header=hdr1).writeto(
        os.path.join(OUT_DIR, "cont_signal_map.fits"), overwrite=True
    )
    fits.PrimaryHDU(noise.astype(np.float32), header=hdr1).writeto(
        os.path.join(OUT_DIR, "cont_noise_map.fits"), overwrite=True
    )

    # Select spaxels for binning: finite S/N, signal positive, S/N >= floor
    mask = np.isfinite(sn) & np.isfinite(signal) & np.isfinite(noise)
    mask &= (signal > 0) & (sn >= SN_FLOOR_VORONOI)
    print(f"Spaxels passing S/N floor ({SN_FLOOR_VORONOI}): {mask.sum()} / {ny*nx}")

    yy, xx = np.where(mask)
    sn_lin = sn[yy, xx]
    sig_lin = signal[yy, xx]
    noise_lin = noise[yy, xx]

    print(f"Running Voronoi binning, target S/N = {TARGET_SN_BIN} ...")
    bin_num, x_node, y_node, x_bar, y_bar, sn_bin, n_pix, scale = voronoi_2d_binning(
        xx.astype(float), yy.astype(float),
        sig_lin, noise_lin,
        TARGET_SN_BIN, plot=False, quiet=True, pixelsize=1,
    )
    n_bins = int(bin_num.max()) + 1
    print(f"  → {n_bins} Voronoi bins")
    print(f"  median S/N achieved per bin = {np.median(sn_bin):.2f}")

    # Build bin map (-1 = unbinned/excluded)
    bin_map = np.full((ny, nx), -1, dtype=np.int32)
    bin_map[yy, xx] = bin_num

    fits.PrimaryHDU(bin_map, header=hdr1).writeto(
        os.path.join(OUT_DIR, "voronoi_bin_map.fits"), overwrite=True
    )
    np.savez(
        os.path.join(OUT_DIR, "voronoi_bins.npz"),
        bin_num=bin_num, xx=xx, yy=yy,
        x_node=x_node, y_node=y_node,
        x_bar=x_bar, y_bar=y_bar,
        sn_bin=sn_bin, n_pix=n_pix, scale=scale,
        wave=wave,
    )
    print("Saved:")
    print("  outputs/voronoi_bin_map.fits")
    print("  outputs/voronoi_bins.npz")
    print("  outputs/cont_SN_map.fits, cont_signal_map.fits, cont_noise_map.fits")

    hdul.close()


if __name__ == "__main__":
    main()
