#!/usr/bin/env python3
"""Interactive Voronoi-bin visualization for a processed galaxy.

Hover over the bin map to highlight bins. Click a bin to open a detail figure
showing the full spectrum (original, continuum-subtracted, stellar model) and
per-line zoom-in panels with Gaussian emission-line fits.

Usage:
    python3 scripts/06_interactive_viz.py <galaxy_name>
    python3 scripts/06_interactive_viz.py ESO154-023
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("QtAgg")  # interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from astropy.io import fits

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, LINES, C_KMS, muse_lsf_fwhm

DOUBLET_RATIOS = {
    ("OIII5007", "OIII4959"): 2.98,
    ("NII6584", "NII6548"):    2.94,
}
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# Build line order + linking
LINE_ORDER = list(LINES.keys())
LINE_WAVES = np.array([LINES[n] for n in LINE_ORDER])
FREE_LINES = []
LINKED = {}
for (lead, follow), r in DOUBLET_RATIOS.items():
    LINKED[follow] = (lead, 1.0 / r)
for ln in LINE_ORDER:
    if ln not in LINKED:
        FREE_LINES.append(ln)
FREE_IDX = {ln: i for i, ln in enumerate(FREE_LINES)}


def gaussian_model(lam, v_kms, sig_kms, amps_dict):
    """Reconstruct the full emission-line model at wavelengths `lam`."""
    z = v_kms / C_KMS
    out = np.zeros_like(lam)
    for name, lam0 in LINES.items():
        lam_obs = lam0 * (1.0 + z)
        A = amps_dict.get(name, 0)
        sig_lam_gas = (sig_kms / C_KMS) * lam_obs
        fwhm_inst = muse_lsf_fwhm(lam_obs)
        sig_lam_inst = fwhm_inst * FWHM_TO_SIGMA
        sig_lam = np.sqrt(sig_lam_gas ** 2 + sig_lam_inst ** 2)
        out += A * np.exp(-0.5 * ((lam - lam_obs) / sig_lam) ** 2)
    return out


def single_line_gaussian(lam, name, lam0, v_kms, sig_kms, amp):
    """Gaussian for a single line at wavelengths `lam`."""
    z = v_kms / C_KMS
    lam_obs = lam0 * (1.0 + z)
    sig_lam_gas = (sig_kms / C_KMS) * lam_obs
    fwhm_inst = muse_lsf_fwhm(lam_obs)
    sig_lam_inst = fwhm_inst * FWHM_TO_SIGMA
    sig_lam = np.sqrt(sig_lam_gas ** 2 + sig_lam_inst ** 2)
    return amp * np.exp(-0.5 * ((lam - lam_obs) / sig_lam) ** 2)


def load_galaxy(galaxy_name):
    """Load all bin-level outputs for a galaxy."""
    out_dir = os.path.join(ROOT, "outputs", galaxy_name)

    bin_map = fits.getdata(os.path.join(out_dir, "voronoi_bin_map.fits"))

    bs = np.load(os.path.join(out_dir, "bin_spectra.npz"))
    wave = bs["wave"]
    bin_spec_orig = bs["bin_spec_orig"]
    bin_spec_sub = bs["bin_spec_sub"]
    stellar_models = bs["stellar_models"]

    bk = np.load(os.path.join(out_dir, "bin_kinematics.npz"), allow_pickle=True)
    v_arr = bk["v_kms"]
    sig_arr = bk["sig_kms"]
    fluxes = bk["fluxes"]  # (N_LINES, n_bins)
    ferrs = bk["flux_errs"]
    line_names = list(bk["line_names"])

    return bin_map, wave, bin_spec_orig, bin_spec_sub, stellar_models, \
        v_arr, sig_arr, fluxes, ferrs, line_names


class BinExplorer:
    """Interactive Voronoi bin explorer."""

    def __init__(self, galaxy_name):
        self.galaxy = galaxy_name
        (self.bin_map, self.wave, self.spec_orig, self.spec_sub,
         self.stellar, self.v_arr, self.sig_arr,
         self.fluxes, self.ferrs, self.line_names) = load_galaxy(galaxy_name)

        self.n_bins = int(self.bin_map.max()) + 1
        self.lmap = {n: i for i, n in enumerate(self.line_names)}
        print(f"Loaded {galaxy_name}: {self.n_bins} bins, "
              f"shape={self.bin_map.shape}, wave=[{self.wave[0]:.0f}, {self.wave[-1]:.0f}] Å")

        # Precompute bin centroids for labeling
        self.bin_centroids = {}
        for b in range(self.n_bins):
            ys, xs = np.where(self.bin_map == b)
            if len(ys) > 0:
                self.bin_centroids[b] = (xs.mean(), ys.mean())

        # Create main figure
        self.fig_main, (self.ax_map, self.ax_info) = plt.subplots(
            1, 2, figsize=(12, 6),
            gridspec_kw={"width_ratios": [1.2, 1]},
        )
        self.fig_main.canvas.manager.set_window_title(f"{galaxy_name} — Bin Explorer")

        # Bin map display
        self._draw_bin_map()
        self._draw_info_panel()

        # Highlight artist
        self.hl_poly = None
        self.current_bin = -1

        # Events
        self.fig_main.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.fig_main.canvas.mpl_connect("button_press_event", self._on_click)

        plt.tight_layout()
        plt.show()

    def _draw_bin_map(self):
        """Display the Voronoi bin map with bin IDs."""
        ny, nx = self.bin_map.shape
        # Show bin map with discrete colors, -1 = light gray
        display = np.where(self.bin_map >= 0, self.bin_map, np.nan)
        im = self.ax_map.imshow(display, origin="lower", cmap="tab20",
                                 interpolation="nearest", aspect="equal")
        self.ax_map.set_title(f"{self.galaxy} — Voronoi bins ({self.n_bins})", fontsize=10)
        self.ax_map.set_xlabel("x [pixel]", fontsize=8)
        self.ax_map.set_ylabel("y [pixel]", fontsize=8)
        self.ax_map.tick_params(labelsize=7)

        # Overlay bin numbers
        for b, (cx, cy) in self.bin_centroids.items():
            if b % max(1, self.n_bins // 50) == 0:
                self.ax_map.text(cx, cy, str(b), fontsize=4, ha="center", va="center",
                                 color="white", weight="bold", alpha=0.7)
        self.ax_map.set_xlim(0, nx)
        self.ax_map.set_ylim(0, ny)

    def _draw_info_panel(self):
        """Initial info panel text."""
        self.ax_info.axis("off")
        self.ax_info.text(0.5, 0.5,
                          "HOVER over the bin map to highlight a bin.\n"
                          "CLICK a bin to view its spectra & line fits.",
                          transform=self.ax_info.transAxes,
                          ha="center", va="center", fontsize=11,
                          bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.9))
        self.info_text = self.ax_info.text(
            0.05, 0.05, "", transform=self.ax_info.transAxes,
            fontsize=8, va="bottom", fontfamily="monospace",
        )

    def _on_hover(self, event):
        """Highlight the bin under the cursor."""
        if event.inaxes != self.ax_map:
            return
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x >= self.bin_map.shape[1] or y < 0 or y >= self.bin_map.shape[0]:
            return
        b = self.bin_map[y, x]
        if b < 0 or b == self.current_bin:
            return

        self.current_bin = b
        # Remove old highlight
        if self.hl_poly is not None and self.hl_poly in self.ax_map.collections:
            self.hl_poly.remove()
            self.hl_poly = None

        # Draw new highlight
        mask = self.bin_map == b
        yy, xx = np.where(mask)
        if len(yy) > 0:
            self.hl_poly = self.ax_map.scatter(
                xx, yy, s=0.5, c="red", alpha=0.3,
                edgecolors="none", zorder=10,
            )

        # Update info
        v = self.v_arr[b] if np.isfinite(self.v_arr[b]) else np.nan
        sig = self.sig_arr[b] if np.isfinite(self.sig_arr[b]) else np.nan
        n_pix = mask.sum()
        info_str = (f"Bin {b}  ({n_pix} spaxels)\n"
                    f"v = {v:+.1f} km/s\n"
                    f"σ = {sig:.1f} km/s")
        self.info_text.set_text(info_str)
        self.fig_main.canvas.draw_idle()

    def _on_click(self, event):
        """Open detail figure showing spectra and per-line fits."""
        if event.inaxes != self.ax_map:
            return
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x >= self.bin_map.shape[1] or y < 0 or y >= self.bin_map.shape[0]:
            return
        b = self.bin_map[y, x]
        if b < 0:
            return
        self._show_bin_detail(b)

    def _show_bin_detail(self, b):
        """Create a detail figure for bin `b`."""
        v = self.v_arr[b]
        sig = self.sig_arr[b]
        if not np.isfinite(v) or not np.isfinite(sig):
            print(f"Bin {b}: no valid fit")
            return

        n_pix = (self.bin_map == b).sum()

        # Build amplitudes dict from fit
        amps_dict = {}
        for name in self.line_names:
            i = self.lmap[name]
            flux = self.fluxes[i, b]
            if np.isfinite(flux):
                lam0 = LINES[name]
                lam_obs = lam0 * (1.0 + v / C_KMS)
                sig_lam_gas = (sig / C_KMS) * lam_obs
                fwhm_inst = muse_lsf_fwhm(lam_obs)
                sig_lam_inst = fwhm_inst * FWHM_TO_SIGMA
                sig_lam = np.sqrt(sig_lam_gas ** 2 + sig_lam_inst ** 2)
                amps_dict[name] = flux / (np.sqrt(2.0 * np.pi) * sig_lam)
            else:
                amps_dict[name] = 0.0

        # Shorten wavelength range to fit window for overview
        i_lo = int(np.searchsorted(self.wave, 4700.0))
        i_hi = int(np.searchsorted(self.wave, 7200.0))
        w_full = self.wave[i_lo:i_hi]
        spec_orig_full = self.spec_orig[i_lo:i_hi, b]
        spec_sub_full = self.spec_sub[i_lo:i_hi, b]
        stellar_full = self.stellar[i_lo:i_hi, b]

        # Emission-line model over full range
        model_full = gaussian_model(w_full, v, sig, amps_dict)

        # Create detail figure
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        fig.canvas.manager.set_window_title(
            f"{self.galaxy} — Bin {b} ({n_pix} spx, v={v:.0f}, σ={sig:.0f})"
        )

        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.25)

        # ----- Top row: full spectrum overview (spans 4 columns, 1 row) -----
        ax_spec = fig.add_subplot(gs[0, :])
        ax_spec.plot(w_full, spec_orig_full, lw=0.5, color="gray", alpha=0.6, label="Original")
        ax_spec.plot(w_full, stellar_full, lw=0.8, color="red", alpha=0.7, label="Stellar continuum")
        ax_spec.plot(w_full, spec_sub_full, lw=0.8, color="black", label="Continuum-subtracted")
        ax_spec.plot(w_full, model_full, lw=0.8, color="blue", alpha=0.7,
                     label="Emission-line fit")
        ax_spec.set_xlim(w_full[0], w_full[-1])
        ax_spec.set_xlabel("Wavelength [Å]", fontsize=8)
        ax_spec.set_ylabel("Flux", fontsize=8)
        ax_spec.set_title(f"Bin {b} — Full spectrum ({n_pix} spaxels)", fontsize=9, fontweight="bold")
        ax_spec.legend(fontsize=6, loc="upper right", ncol=4)
        ax_spec.tick_params(labelsize=6)
        ax_spec.minorticks_on()

        # ----- Per-line zoom panels (3 rows × 4 columns = 12 lines) -----
        detected = []
        for name in self.line_names:
            i = self.lmap[name]
            f = self.fluxes[i, b]
            e = self.ferrs[i, b]
            sn = f / e if (e > 0 and np.isfinite(e)) else 0
            detected.append(sn >= 3.0)

        for idx, name in enumerate(self.line_names):
            row = 1 + idx // 4
            col = idx % 4
            ax = fig.add_subplot(gs[row, col])

            lam0 = LINES[name]
            lam_obs = lam0 * (1.0 + v / C_KMS)
            win_hw = 25.0

            w_mask = (self.wave > lam_obs - win_hw) & (self.wave < lam_obs + win_hw)
            w_zoom = self.wave[w_mask]
            if len(w_zoom) == 0:
                ax.text(0.5, 0.5, "out of range", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7)
                ax.set_title(name, fontsize=7)
                continue

            sub_zoom = self.spec_sub[w_mask, b]
            model_zoom = gaussian_model(w_zoom, v, sig, amps_dict)
            single_zoom = single_line_gaussian(w_zoom, name, lam0, v, sig,
                                              amps_dict.get(name, 0))

            # Plot
            ax.step(w_zoom, sub_zoom, lw=0.6, color="black", where="mid", label="Cont-sub")
            ax.plot(w_zoom, model_zoom, lw=0.8, color="blue", alpha=0.5,
                    label="Full model")
            ax.plot(w_zoom, single_zoom, lw=1.0, color="red", alpha=0.8,
                    label="This line")

            # Line info
            i = self.lmap[name]
            flux = self.fluxes[i, b]
            ferr = self.ferrs[i, b]
            sn = flux / ferr if (ferr > 0 and np.isfinite(ferr)) else 0

            # Mark S/N
            detected_str = f" S/N={sn:.1f}" if detected[idx] else f" S/N={sn:.1f} <3"
            color = "darkgreen" if detected[idx] else "darkred"

            ax.axvline(lam_obs, color=color, lw=0.5, ls="--", alpha=0.5)
            ax.set_title(f"{name} {LINES[name]:.1f}Å", fontsize=6.5, fontweight="bold")
            ax.set_xlabel(f"{detected_str}", fontsize=5.5, color=color)
            ax.tick_params(labelsize=5)
            ax.set_xlim(lam_obs - win_hw, lam_obs + win_hw)

            # Simple y-lim based on data
            y_data = sub_zoom[np.isfinite(sub_zoom)]
            if len(y_data) > 0:
                y_max = np.percentile(np.abs(y_data), 95) * 1.3
                y_max = max(y_max, 0.001)
                ax.set_ylim(-y_max, y_max)

        plt.show(block=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/06_interactive_viz.py <galaxy_name>")
        print("Example: python3 scripts/06_interactive_viz.py ESO154-023")
        sys.exit(1)

    galaxy_name = sys.argv[1]
    out_dir = os.path.join(ROOT, "outputs", galaxy_name)

    if not os.path.isdir(out_dir):
        print(f"Error: no output directory for {galaxy_name} at {out_dir}")
        sys.exit(1)

    required = ["voronoi_bin_map.fits", "bin_spectra.npz", "bin_kinematics.npz"]
    missing = [f for f in required
               if not os.path.exists(os.path.join(out_dir, f))]
    if missing:
        print(f"Error: missing files for {galaxy_name}: {missing}")
        print("Run the bin-level pipeline first (stages 1-4).")
        sys.exit(1)

    explorer = BinExplorer(galaxy_name)


if __name__ == "__main__":
    main()
