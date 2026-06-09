"""Stage 5 (bin-level): N-BPT, S-BPT, O-BPT diagrams at Voronoi-bin resolution.

Uses bin-level line fluxes from stage 4, applies S/N >= 3 in all required lines,
classifies bins with standard demarcation lines, and produces a multi-page PDF
with two-column layout (BPT scatter + spatial map at bin resolution).
"""

import os
import sys
import numpy as np
from astropy.io import fits
from scipy.optimize import brentq

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT_DIR, GALAXY, ROOT, SN_DETECT

LINE_DIR = os.path.join(OUT_DIR, "bin_line_maps")
BPT_DIR = os.path.join(ROOT, "outputs", "bpt")
os.makedirs(BPT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Demarcation-line functions
# ---------------------------------------------------------------------------
def kauffmann03(x):
    with np.errstate(divide="ignore"):
        return 0.61 / (x - 0.05) + 1.30


def kewley01(x):
    with np.errstate(divide="ignore"):
        return 0.61 / (x - 0.47) + 1.19


def kewley06_sf_s2(x):
    with np.errstate(divide="ignore"):
        return 0.72 / (x - 0.32) + 1.30


def kewley06_syliner_s2(x):
    return 1.89 * x + 0.76


def kewley06_sf_o1(x):
    with np.errstate(divide="ignore"):
        return 0.73 / (x + 0.59) + 1.33


def kewley06_syliner_o1(x):
    return 1.18 * x + 1.30


# Crossing point of Ka03 and Ke01 (N-BPT)
def _diff_ka_ke(x):
    return kauffmann03(x) - kewley01(x)


try:
    X_CROSS_NBPT = brentq(_diff_ka_ke, -2.0, -0.5)
except Exception:
    X_CROSS_NBPT = -1.28
print(f"Ka03 and Ke01 cross at x = {X_CROSS_NBPT:.3f}")


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
CLR_SF      = "#3B82F6"
CLR_INTERM  = "#A855F7"
CLR_AGN     = "#22C55E"
CLR_LINER   = "#EF4444"
CLR_BG      = "#e0e0e0"
CLR_KE01    = "#111827"
CLR_KA03    = "#6366F1"
CLR_KE06_SF = "#111827"
CLR_KE06_SL = "#F97316"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_nbpt(n2ha, o3hb):
    cls = np.full_like(n2ha, -1, dtype=int)
    k03 = kauffmann03(n2ha)
    k01 = kewley01(n2ha)
    finite_k03 = np.isfinite(k03)
    finite_k01 = np.isfinite(k01)
    k03_exists = (n2ha > X_CROSS_NBPT) & finite_k03
    sf = (k03_exists & (o3hb <= k03)) | ((~k03_exists) & finite_k01 & (o3hb <= k01))
    cls[sf] = 0
    inter = k03_exists & ~sf & finite_k01 & (o3hb <= k01)
    cls[inter] = 1
    agn = finite_k01 & (o3hb > k01)
    cls[agn] = 2
    return cls


def classify_sbpt(s2ha, o3hb):
    cls = np.full_like(s2ha, -1, dtype=int)
    k06_sf = kewley06_sf_s2(s2ha)
    k06_sl = kewley06_syliner_s2(s2ha)
    finite_sf = np.isfinite(k06_sf)
    sf = ~finite_sf | (o3hb <= k06_sf)
    cls[sf] = 0
    non_sf = finite_sf & ~sf
    cls[non_sf & (o3hb >= k06_sl)] = 2
    cls[non_sf & (o3hb < k06_sl)] = 3
    return cls


def classify_obpt(o1ha, o3hb):
    cls = np.full_like(o1ha, -1, dtype=int)
    k06_sf = kewley06_sf_o1(o1ha)
    k06_sl = kewley06_syliner_o1(o1ha)
    finite_sf = np.isfinite(k06_sf)
    sf = ~finite_sf | (o3hb <= k06_sf)
    cls[sf] = 0
    non_sf = finite_sf & ~sf
    cls[non_sf & (o3hb >= k06_sl)] = 2
    cls[non_sf & (o3hb < k06_sl)] = 3
    return cls


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_demarcation_lines(ax, bpt_type, xlim, ylim):
    if bpt_type == "n":
        x_ke = np.linspace(xlim[0], 0.46, 500)
        y_ke = kewley01(x_ke)
        mask = (y_ke > ylim[0]) & (y_ke < ylim[1])
        ax.plot(x_ke[mask], y_ke[mask], color=CLR_KE01, lw=1.8, ls="-",
                label="Kewley+01 (max SF)", zorder=3)

        x_ka = np.linspace(X_CROSS_NBPT + 0.01, -0.001, 500)
        y_ka = kauffmann03(x_ka)
        mask = (y_ka > ylim[0]) & (y_ka < ylim[1])
        ax.plot(x_ka[mask], y_ka[mask], color=CLR_KA03, lw=1.8, ls="--",
                label="Kauffmann+03 (emp. SF)", zorder=3)

        xfine = np.linspace(xlim[0], xlim[1], 800)
        x_sf = xfine[xfine < 0.0]
        y_upper_sf = np.where(x_sf > X_CROSS_NBPT, kauffmann03(x_sf), kewley01(x_sf))
        ax.fill_between(x_sf, ylim[0], np.clip(y_upper_sf, ylim[0], ylim[1]),
                        color=CLR_SF, alpha=0.06, zorder=0)
        x_co = xfine[(xfine > X_CROSS_NBPT) & (xfine < 0.0)]
        ax.fill_between(x_co,
                        np.clip(kauffmann03(x_co), ylim[0], ylim[1]),
                        np.clip(kewley01(x_co), ylim[0], ylim[1]),
                        color=CLR_INTERM, alpha=0.06, zorder=0)
        x_agn = xfine[(xfine >= -0.5) & (xfine <= xlim[1])]
        y_ke_agn = np.where(x_agn < 0.46, kewley01(x_agn), ylim[0])
        ax.fill_between(x_agn, np.clip(y_ke_agn, ylim[0], ylim[1]), ylim[1],
                        where=(y_ke_agn < ylim[1]),
                        color=CLR_AGN, alpha=0.04, zorder=0)
        ax.text(-1.05, -0.5, "Star\nForming", fontsize=8, fontweight="bold",
                color=CLR_SF, ha="center", alpha=0.85)
        ax.text(-0.25, 0.35, "Comp.", fontsize=8, fontweight="bold",
                color=CLR_INTERM, ha="center", alpha=0.85)
        ax.text(0.25, 1.0, "AGN", fontsize=9, fontweight="bold",
                color=CLR_AGN, ha="center", alpha=0.85)

    elif bpt_type == "s":
        x_sf = np.linspace(xlim[0], 0.319, 500)
        y_sf = kewley06_sf_s2(x_sf)
        mask = (y_sf > ylim[0]) & (y_sf < ylim[1])
        ax.plot(x_sf[mask], y_sf[mask], color=CLR_KE06_SF, lw=1.8, ls="-",
                label="Kewley+06 (max SF)", zorder=3)
        x_sl = np.linspace(max(xlim[0], -0.3), min(xlim[1], 0.6), 200)
        y_sl = kewley06_syliner_s2(x_sl)
        mask = (y_sl > ylim[0]) & (y_sl < ylim[1])
        ax.plot(x_sl[mask], y_sl[mask], color=CLR_KE06_SL, lw=1.8, ls="-.",
                label="Kewley+06 (Sy/LINER)", zorder=3)
        xfine = np.linspace(xlim[0], xlim[1], 600)
        x_sf_fill = xfine[xfine < 0.32]
        ax.fill_between(x_sf_fill, ylim[0],
                        np.clip(kewley06_sf_s2(x_sf_fill), ylim[0], ylim[1]),
                        color=CLR_SF, alpha=0.06, zorder=0)
        x_sy = xfine[(xfine > -0.3) & (xfine < 0.32)]
        ax.fill_between(x_sy,
                        np.clip(kewley06_syliner_s2(x_sy), ylim[0], ylim[1]), ylim[1],
                        where=(kewley06_syliner_s2(x_sy) < ylim[1]),
                        color=CLR_AGN, alpha=0.04, zorder=0)
        ax.fill_between(x_sy,
                        np.clip(kewley06_sf_s2(x_sy), ylim[0], ylim[1]),
                        np.clip(kewley06_syliner_s2(x_sy), ylim[0], ylim[1]),
                        color=CLR_LINER, alpha=0.04, zorder=0)
        ax.text(-0.9, -0.6, "Star\nForming", fontsize=8, fontweight="bold",
                color=CLR_SF, ha="center", alpha=0.85)
        ax.text(0.15, 1.1, "Seyfert", fontsize=8, fontweight="bold",
                color=CLR_AGN, ha="center", alpha=0.85)
        ax.text(0.45, -0.25, "LINER /\nShocks", fontsize=7, fontweight="bold",
                color=CLR_LINER, ha="center", alpha=0.85)

    elif bpt_type == "o":
        x_sf = np.linspace(xlim[0], -0.591, 500)
        y_sf = kewley06_sf_o1(x_sf)
        mask = (y_sf > ylim[0]) & (y_sf < ylim[1])
        ax.plot(x_sf[mask], y_sf[mask], color=CLR_KE06_SF, lw=1.8, ls="-",
                label="Kewley+06 (max SF)", zorder=3)
        x_sl = np.linspace(max(xlim[0], -1.12), min(xlim[1], 0.1), 200)
        y_sl = kewley06_syliner_o1(x_sl)
        mask = (y_sl > ylim[0]) & (y_sl < ylim[1])
        ax.plot(x_sl[mask], y_sl[mask], color=CLR_KE06_SL, lw=1.8, ls="-.",
                label="Kewley+06 (Sy/LINER)", zorder=3)
        xfine = np.linspace(xlim[0], xlim[1], 600)
        x_sf_fill = xfine[xfine < -0.59]
        ax.fill_between(x_sf_fill, ylim[0],
                        np.clip(kewley06_sf_o1(x_sf_fill), ylim[0], ylim[1]),
                        color=CLR_SF, alpha=0.06, zorder=0)
        x_non = xfine[(xfine > -1.12) & (xfine < -0.59)]
        ax.fill_between(x_non,
                        np.clip(kewley06_syliner_o1(x_non), ylim[0], ylim[1]), ylim[1],
                        where=(kewley06_syliner_o1(x_non) < ylim[1]),
                        color=CLR_AGN, alpha=0.04, zorder=0)
        ax.fill_between(x_non,
                        np.clip(kewley06_sf_o1(x_non), ylim[0], ylim[1]),
                        np.clip(kewley06_syliner_o1(x_non), ylim[0], ylim[1]),
                        color=CLR_LINER, alpha=0.04, zorder=0)
        ax.text(-2.0, -0.6, "Star\nForming", fontsize=8, fontweight="bold",
                color=CLR_SF, ha="center", alpha=0.85)
        ax.text(-0.4, 1.1, "Seyfert", fontsize=8, fontweight="bold",
                color=CLR_AGN, ha="center", alpha=0.85)
        ax.text(-0.2, -0.25, "LINER /\nShocks", fontsize=7, fontweight="bold",
                color=CLR_LINER, ha="center", alpha=0.85)


def scatter_bpt(ax, x, y, cls):
    colors = {0: CLR_SF, 1: CLR_INTERM, 2: CLR_AGN, 3: CLR_LINER}
    labels = {0: "SF", 1: "Intermediate", 2: "Seyfert", 3: "LINER"}
    ax.scatter(x, y, s=0.8, c=CLR_BG, edgecolors="none", alpha=0.25, zorder=0)
    for code in sorted(set(np.unique(cls)) - {-1}):
        mask = cls == code
        if mask.sum() == 0:
            continue
        ax.scatter(x[mask], y[mask], s=3.0, c=colors[code],
                   edgecolors="none", label=labels[code], zorder=1)
    ax.legend(loc="upper left", fontsize=5.5, markerscale=2.5,
              frameon=True, fancybox=False, edgecolor="gray")


def draw_spatial_map(ax, cls_2d, ha_2d, title):
    """Draw spatial map of BPT classifications expanded from bin to pixel grid."""
    from matplotlib.colors import to_rgb
    ny, nx = cls_2d.shape
    rgb = np.zeros((ny, nx, 3))
    color_map = {0: to_rgb(CLR_SF), 1: to_rgb(CLR_INTERM),
                 2: to_rgb(CLR_AGN), 3: to_rgb(CLR_LINER)}
    bg_color = np.array([0.95, 0.95, 0.95])
    for code, clr in color_map.items():
        rgb[cls_2d == code] = clr
    rgb[cls_2d == -1] = bg_color
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ha_valid = np.where(np.isfinite(ha_2d) & (ha_2d > 0), ha_2d, 0)
    if ha_valid.max() > 0:
        levels = np.unique(np.percentile(ha_valid[ha_valid > 0], [50, 80, 95]))
        if len(levels) >= 2:
            ax.contour(ha_valid, levels=levels, colors="black", linewidths=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [pixel]", fontsize=7)
    ax.set_ylabel("y [pixel]", fontsize=7)
    ax.tick_params(labelsize=6)


def expand_bin_to_2d(bin_values, bin_map, n_bins):
    """Expand a 1D per-bin array to a 2D (ny, nx) map using the bin map."""
    ny, nx = bin_map.shape
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    for b in range(n_bins):
        mask_2d = bin_map == b
        if mask_2d.any() and np.isfinite(bin_values[b]):
            out[mask_2d] = bin_values[b]
    return out


def main():
    print("Loading bin kinematics ...")
    bk = np.load(os.path.join(OUT_DIR, "bin_kinematics.npz"), allow_pickle=True)
    fluxes = bk["fluxes"]      # (N_LINES, n_bins)
    ferrs = bk["flux_errs"]
    line_names = list(bk["line_names"])
    n_bins = fluxes.shape[1]

    # Map line names to indices
    lmap = {n: i for i, n in enumerate(line_names)}
    print(f"  {n_bins} bins, {len(line_names)} lines: {line_names}")

    # Load Voronoi bin map for spatial plots
    bin_map = fits.getdata(os.path.join(OUT_DIR, "voronoi_bin_map.fits"))
    bin_map = bin_map.astype(int)
    ny, nx = bin_map.shape

    # Extract line fluxes (S/N >= 3)
    def flux_sn3(name):
        i = lmap[name]
        f = fluxes[i].copy()
        e = ferrs[i].copy()
        sn = np.where(e > 0, f / e, 0.0)
        f[sn < SN_DETECT] = np.nan
        return f

    ha  = flux_sn3("Halpha")
    hb  = flux_sn3("Hbeta")
    n2  = flux_sn3("NII6584")
    o3  = flux_sn3("OIII5007")
    s2_6716 = flux_sn3("SII6716")
    s2_6731 = flux_sn3("SII6731")
    o1  = flux_sn3("OI6300")
    s2  = s2_6716 + s2_6731

    # Combined S/N >= 3 masks for each BPT
    valid_n = np.isfinite(ha) & np.isfinite(hb) & np.isfinite(n2) & np.isfinite(o3)
    valid_s = np.isfinite(ha) & np.isfinite(hb) & np.isfinite(s2) & np.isfinite(o3)
    valid_o = np.isfinite(ha) & np.isfinite(hb) & np.isfinite(o1) & np.isfinite(o3)
    print(f"N-BPT valid bins: {valid_n.sum()}")
    print(f"S-BPT valid bins: {valid_s.sum()}")
    print(f"O-BPT valid bins: {valid_o.sum()}")

    # Line ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        log_n2_ha = np.where(valid_n, np.log10(n2 / ha), np.nan)
        log_o3_hb = np.where(valid_n, np.log10(o3 / hb), np.nan)
        log_s2_ha = np.where(valid_s, np.log10(s2 / ha), np.nan)
        log_o3_hb_s = np.where(valid_s, np.log10(o3 / hb), np.nan)
        log_o1_ha = np.where(valid_o, np.log10(o1 / ha), np.nan)
        log_o3_hb_o = np.where(valid_o, np.log10(o3 / hb), np.nan)

    # Classify
    print("Classifying bins ...")
    cls_n = np.full(n_bins, -1, dtype=int)
    cls_s = np.full(n_bins, -1, dtype=int)
    cls_o = np.full(n_bins, -1, dtype=int)
    if valid_n.sum() > 0:
        cls_n[valid_n] = classify_nbpt(log_n2_ha[valid_n], log_o3_hb[valid_n])
    if valid_s.sum() > 0:
        cls_s[valid_s] = classify_sbpt(log_s2_ha[valid_s], log_o3_hb_s[valid_s])
    if valid_o.sum() > 0:
        cls_o[valid_o] = classify_obpt(log_o1_ha[valid_o], log_o3_hb_o[valid_o])

    # Statistics
    for name, cls_arr, valid in [("N-BPT", cls_n, valid_n),
                                  ("S-BPT", cls_s, valid_s),
                                  ("O-BPT", cls_o, valid_o)]:
        codes = {0: "SF", 1: "Intermediate", 2: "Seyfert", 3: "LINER"}
        print(f"\n{name}:")
        n_tot = valid.sum()
        for k, v in codes.items():
            n = (cls_arr[valid] == k).sum()
            if n_tot > 0:
                print(f"  {v}: {n} ({100*n/n_tot:.1f}%)")

    # ---------- Expand to 2D for spatial maps ---------------------------------
    ha_2d = expand_bin_to_2d(ha, bin_map, n_bins)

    def _safe_2d_class(cls_1d):
        """Expand classification to 2D, filling NaN in output where bin has no value."""
        out = expand_bin_to_2d(cls_1d.astype(float), bin_map, n_bins)
        out[np.isnan(out)] = -1
        return out.astype(int)

    cls_n_2d = _safe_2d_class(cls_n)
    cls_s_2d = _safe_2d_class(cls_s)
    cls_o_2d = _safe_2d_class(cls_o)

    # ---------- PDF -----------------------------------------------------------
    print("\nPlotting ...")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch
    out_pdf = os.path.join(BPT_DIR, f"{GALAXY}_bpt.pdf")

    BPT_CONFIG = {
        "n": dict(xlabel=r"$\log\ [\mathrm{NII}]\lambda6584\ /\ \mathrm{H}\alpha$",
                  title="N-BPT", xlim=(-1.5, 0.6), ylim=(-1.2, 1.5)),
        "s": dict(xlabel=r"$\log\ [\mathrm{SII}]\lambda\lambda6716,6731\ /\ \mathrm{H}\alpha$",
                  title="S-BPT", xlim=(-1.5, 0.8), ylim=(-1.2, 1.5)),
        "o": dict(xlabel=r"$\log\ [\mathrm{OI}]\lambda6300\ /\ \mathrm{H}\alpha$",
                  title="O-BPT", xlim=(-2.8, 0.2), ylim=(-1.2, 1.5)),
    }

    data_map = {
        "n": (log_n2_ha, log_o3_hb, cls_n, valid_n, cls_n_2d),
        "s": (log_s2_ha, log_o3_hb_s, cls_s, valid_s, cls_s_2d),
        "o": (log_o1_ha, log_o3_hb_o, cls_o, valid_o, cls_o_2d),
    }

    with PdfPages(out_pdf) as pdf:
        for bpt_type, cfg in BPT_CONFIG.items():
            x_data, y_data, cls_1d, valid, cls_2d = data_map[bpt_type]
            xlim, ylim = cfg["xlim"], cfg["ylim"]

            fig = plt.figure(figsize=(9, 4.5), constrained_layout=True)
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.25)

            # Left: BPT scatter
            ax_left = fig.add_subplot(gs[0])
            ax_left.set_facecolor("white")
            xx = x_data[valid]
            yy = y_data[valid]
            cc = cls_1d[valid]
            scatter_bpt(ax_left, xx, yy, cc)
            plot_demarcation_lines(ax_left, bpt_type, xlim, ylim)
            ax_left.set_xlim(xlim)
            ax_left.set_ylim(ylim)
            ax_left.set_xlabel(cfg["xlabel"], fontsize=8)
            ax_left.set_ylabel(r"$\log\ [\mathrm{OIII}]\lambda5007\ /\ \mathrm{H}\beta$", fontsize=8)
            ax_left.set_title(f"{cfg['title']} ({valid.sum()} bins)", fontsize=9)
            ax_left.tick_params(labelsize=7)
            ax_left.minorticks_on()

            # Right: spatial classification map
            ax_right = fig.add_subplot(gs[1])
            # Build valid mask expanded to 2D for display
            valid_2d = expand_bin_to_2d(valid.astype(float), bin_map, n_bins)
            cls_display_2d = np.where(valid_2d > 0, cls_2d, -1)
            draw_spatial_map(ax_right, cls_display_2d, ha_2d, f"{cfg['title']} — spatial")

            legend_elements = [
                Patch(facecolor=CLR_SF, label="SF"),
                Patch(facecolor=CLR_AGN, label="Seyfert"),
                Patch(facecolor=CLR_LINER, label="LINER/Shock"),
            ]
            if bpt_type == "n":
                legend_elements.insert(1, Patch(facecolor=CLR_INTERM, label="Intermediate"))
            ax_right.legend(handles=legend_elements, loc="upper right",
                            fontsize=5, frameon=True, fancybox=False,
                            edgecolor="gray", ncol=1)

            pdf.savefig(fig, dpi=200, bbox_inches="tight")
            plt.close(fig)

    print(f"\nSaved {out_pdf}")
    print("Done.")


if __name__ == "__main__":
    main()
