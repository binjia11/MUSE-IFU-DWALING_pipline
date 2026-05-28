"""
BPT Demarcation Lines v3 — matching Cresci et al. (2017) Figure 5
=================================================================
Fixes:
  - Ka03 line truncated where it meets Ke01 (~x = -1.28) so they don't cross
  - No Seyfert/LINER dividing line on the N-BPT (only on S-BPT and O-BPT)
  - Axis ranges adjusted to match typical usage
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ── Colour palette ──────────────────────────────────────────────
COL_SF      = "#3B82F6"
COL_COMP    = "#A855F7"
COL_SEYFERT = "#22C55E"
COL_LINER   = "#EF4444"
COL_KE01    = "#111827"
COL_KA03    = "#6366F1"
COL_KE06    = "#F97316"

# ================================================================
#  DEMARCATION-LINE FUNCTIONS
# ================================================================
def kewley01_NII(x):
    return 0.61 / (x - 0.47) + 1.19

def kauffmann03(x):
    return 0.61 / (x - 0.05) + 1.30

def kewley06_NII_SyLINER(x):
    return 1.89 * x + 0.76

def kewley01_SII(x):
    return 0.72 / (x - 0.32) + 1.30

def kewley06_SII_SyLINER(x):
    return 1.89 * x + 0.76

def kewley01_OI(x):
    return 0.73 / (x + 0.59) + 1.33

def kewley06_OI_SyLINER(x):
    return 1.18 * x + 1.30


# ── Find crossing point of Ka03 and Ke01 ───────────────────────
def diff_ka_ke(x):
    return kauffmann03(x) - kewley01_NII(x)

x_cross = brentq(diff_ka_ke, -2.0, -0.5)
print(f"Ka03 and Ke01 cross at x = {x_cross:.3f}")


# ================================================================
#  PLOTTING
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
fig.subplots_adjust(wspace=0.30)

# ── Panel 1: N-BPT ──────────────────────────────────────────────
ax = axes[0]
xlim, ylim = (-1.5, 0.6), (-1.2, 1.5)

# Kewley+2001 max-starburst (drawn over full valid range)
x_ke = np.linspace(xlim[0], 0.46, 500)  # extend to near the asymptote at 0.47
y_ke = kewley01_NII(x_ke)
mask = (y_ke > ylim[0]) & (y_ke < ylim[1])
ax.plot(x_ke[mask], y_ke[mask], color=COL_KE01, lw=2.2, ls='-',
        label='Kewley+2001 (max SF)', zorder=3)

# Kauffmann+2003 — ONLY from the crossing point to x < 0
# This ensures Ka03 always stays BELOW Ke01
x_ka = np.linspace(x_cross + 0.01, -0.001, 500)
y_ka = kauffmann03(x_ka)
mask = (y_ka > ylim[0]) & (y_ka < ylim[1])
ax.plot(x_ka[mask], y_ka[mask], color=COL_KA03, lw=2.2, ls='--',
        label='Kauffmann+2003 (emp. SF)', zorder=3)

# NO Seyfert/LINER line on N-BPT (matching Cresci+2017)

# Shading
xfine = np.linspace(xlim[0], xlim[1], 800)

# SF (below Ka03, or below Ke01 where Ka03 doesn't exist)
x_sf = xfine[xfine < 0.0]
y_upper_sf = np.where(x_sf > x_cross,
                      kauffmann03(x_sf),
                      kewley01_NII(x_sf))
ax.fill_between(x_sf, ylim[0], np.minimum(y_upper_sf, ylim[1]),
                color=COL_SF, alpha=0.07, zorder=0)

# Composite (between Ka03 and Ke01, only where Ka03 exists)
x_co = xfine[(xfine > x_cross) & (xfine < 0.0)]
y_ka_co = kauffmann03(x_co)
y_ke_co = kewley01_NII(x_co)
ax.fill_between(x_co, y_ka_co, np.minimum(y_ke_co, ylim[1]),
                color=COL_COMP, alpha=0.07, zorder=0)

# AGN zone (above Ke01) — no Sy/LINER split on N-BPT
x_agn = xfine[(xfine >= -0.5) & (xfine <= 0.6)]
y_ke_agn = np.where(x_agn < 0.46, kewley01_NII(x_agn), ylim[0])
ax.fill_between(x_agn, np.maximum(y_ke_agn, ylim[0]), ylim[1],
                where=(y_ke_agn < ylim[1]),
                color=COL_SEYFERT, alpha=0.05, zorder=0)

ax.text(-1.05, -0.6, 'Star\nForming', fontsize=11, fontweight='bold',
        color=COL_SF, ha='center', alpha=0.85)
ax.text(-0.25, 0.30, 'Comp.', fontsize=10, fontweight='bold',
        color=COL_COMP, ha='center', alpha=0.85)
ax.text(0.25, 1.05, 'AGN', fontsize=11, fontweight='bold',
        color=COL_SEYFERT, ha='center', alpha=0.85)

ax.set_xlim(xlim); ax.set_ylim(ylim)
ax.set_xlabel(r'log([N$\,$II]$\lambda$6584 / H$\alpha$)', fontsize=13)
ax.set_ylabel(r'log([O$\,$III]$\lambda$5007 / H$\beta$)', fontsize=13)
ax.set_title('N-BPT', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)


# ── Panel 2: S-BPT ──────────────────────────────────────────────
ax = axes[1]
xlim, ylim = (-1.5, 0.8), (-1.2, 1.5)

# Kewley+2006 max-starburst
x = np.linspace(xlim[0], 0.319, 500)
y = kewley01_SII(x)
mask = (y > ylim[0]) & (y < ylim[1])
ax.plot(x[mask], y[mask], color=COL_KE01, lw=2.2, ls='-',
        label='Kewley+2006 (max SF)', zorder=3)

# Kewley+2006 Seyfert/LINER
x = np.linspace(-0.31, 0.6, 100)
y = kewley06_SII_SyLINER(x)
mask = (y > ylim[0]) & (y < ylim[1])
ax.plot(x[mask], y[mask], color=COL_KE06, lw=2.2, ls='-.',
        label='Kewley+2006 (Sy/LINER)', zorder=3)

ax.text(-0.9, -0.6, 'Star\nForming', fontsize=11, fontweight='bold',
        color=COL_SF, ha='center', alpha=0.85)
ax.text(0.15, 1.2, 'Seyfert', fontsize=10, fontweight='bold',
        color=COL_SEYFERT, ha='center', alpha=0.85)
ax.text(0.45, -0.3, 'LINER /\nShocks', fontsize=10, fontweight='bold',
        color=COL_LINER, ha='center', alpha=0.85)

ax.set_xlim(xlim); ax.set_ylim(ylim)
ax.set_xlabel(r'log([S$\,$II]$\lambda\lambda$6716,30 / H$\alpha$)', fontsize=13)
ax.set_ylabel(r'log([O$\,$III]$\lambda$5007 / H$\beta$)', fontsize=13)
ax.set_title('S-BPT', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)


# ── Panel 3: O-BPT ──────────────────────────────────────────────
ax = axes[2]
xlim, ylim = (-2.8, 0.2), (-1.2, 1.5)

# Kewley+2006 max-starburst
x = np.linspace(xlim[0], -0.591, 500)
y = kewley01_OI(x)
mask = (y > ylim[0]) & (y < ylim[1])
ax.plot(x[mask], y[mask], color=COL_KE01, lw=2.2, ls='-',
        label='Kewley+2006 (max SF)', zorder=3)

# Kewley+2006 Seyfert/LINER
x = np.linspace(-1.12, 0.1, 100)
y = kewley06_OI_SyLINER(x)
mask = (y > ylim[0]) & (y < ylim[1])
ax.plot(x[mask], y[mask], color=COL_KE06, lw=2.2, ls='-.',
        label='Kewley+2006 (Sy/LINER)', zorder=3)

ax.text(-2.0, -0.6, 'Star\nForming', fontsize=11, fontweight='bold',
        color=COL_SF, ha='center', alpha=0.85)
ax.text(-0.4, 1.1, 'Seyfert', fontsize=10, fontweight='bold',
        color=COL_SEYFERT, ha='center', alpha=0.85)
ax.text(-0.2, -0.3, 'LINER /\nShocks', fontsize=10, fontweight='bold',
        color=COL_LINER, ha='center', alpha=0.85)

ax.set_xlim(xlim); ax.set_ylim(ylim)
ax.set_xlabel(r'log([O$\,$I]$\lambda$6300 / H$\alpha$)', fontsize=13)
ax.set_ylabel(r'log([O$\,$III]$\lambda$5007 / H$\beta$)', fontsize=13)
ax.set_title('O-BPT', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)


# ── Global polish ───────────────────────────────────────────────
for a in axes:
    a.tick_params(labelsize=11, direction='in', top=True, right=True)
    a.minorticks_on()
    a.tick_params(which='minor', direction='in', top=True, right=True)

fig.suptitle('BPT Demarcation Lines  (Kewley+2001/2006, Kauffmann+2003)',
             fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('/home/claude/bpt_demarcation_v3.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Done — saved bpt_demarcation_v3.png")
