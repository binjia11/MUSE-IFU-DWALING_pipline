# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MUSE IFU pipeline for studying low-metallicity galaxies with ionized gas shocks in the LMC/DWALIN sample. Performs Voronoi binning, pPXF stellar continuum fitting (BPASS v2.2.1), **bin-level** emission line fitting, and **bin-level** BPT classification of ionization mechanisms (SF vs AGN vs shocks).

**23 galaxies processed** across the DWALIN sample + additional low-metallicity targets.

## Conda environment

```
/opt/anaconda3/envs/uclchem_3.4/bin/python3   (env name: uclchem_3.4)
```

Key packages: `astropy`, `numpy`, `scipy`, `matplotlib`, `ppxf`, `vorbin`.

## Running the pipeline

**Full pipeline for all sample galaxies:**

```bash
# Bin-level pipeline (default) — emission lines + BPT at Voronoi-bin resolution
python3 scripts/run_pipeline.py

# Parallel (N galaxies concurrently, 1 core each)
python3 scripts/run_pipeline.py --parallel        # all cores
python3 scripts/run_pipeline.py --parallel 4      # 4 concurrent galaxies

# Legacy per-spaxel pipeline
python3 scripts/run_pipeline.py --spaxel
```

In parallel mode, each galaxy gets 1 dedicated core (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). v_sys is measured sequentially upfront to avoid memory pressure from loading multiple MUSE cubes simultaneously.

The driver auto-discovers all `.fits` files in `DWALIN_Sample/`, measures systemic velocity via Hα cross-correlation, and skips galaxies with existing `outputs/bpt/<galaxy>_bpt.pdf`. No timeouts — runs each stage to completion (long runtimes are from large Voronoi bin counts, not code issues).

**Individual stage for one galaxy** (from the project root):

```bash
export PIPE_GALAXY=ESO154-023
export PIPE_CUBE_PATH="/Users/binjia/Desktop/low-metallicity_shocks_LMC/DWALIN_Sample/ESO154-023.fits"
export PIPE_OUT_DIR="/Users/binjia/Desktop/low-metallicity_shocks_LMC/outputs/ESO154-023"
export PIPE_V_SYS=581
export PIPE_TEMPLATE_LIB=bpass
python3 scripts/04_fit_emission_lines_bins.py
```

**Interactive visualization** (post-pipeline):

```bash
python3 scripts/06_interactive_viz.py ESO154-023
```

Hover over the bin map to highlight bins; click to view full spectra and per-line zoom-ins with Gaussian fits.

## Architecture

The pipeline has 6 stages (5 pipeline + 1 interactive viz), each a standalone script, communicating through files on disk:

1. **`01_voronoi_binning.py`** — Computes continuum S/N per spaxel (5300–5530 Å rest-frame), then uses `vorbin` to adaptively bin spaxels to S/N ≥ 30. Outputs `voronoi_bin_map.fits` + `voronoi_bins.npz`.

2. **`02_ppxf_fit_bins.py`** — For each Voronoi bin: sums spaxel spectra, log-rebins, runs pPXF with **BPASS v2.2.1** SSP templates (binary, 1 Myr–100 Gyr, 8 metallicities z001–z020) + Gaussian gas templates (3 kinematic components: stars, Balmer, forbidden). Fits over 4760–7400 Å observed-frame. Stores stellar continuum models and gas models per bin in `ppxf_bin_fits.npz`. Set `PIPE_TEMPLATE_LIB=emiles` to use EMILES instead.

3. **`03_subtract_continuum_bins.py`** (bin-level, default) — Integrates spaxel spectra per bin, subtracts the stage-2 stellar model directly (no per-spaxel rescaling since the model was fit on the same spectrum). Applies local 3rd-order polynomial refinement around each emission line (±60 Å window, masking line core ±12 Å). Outputs `bin_spectra.npz` (original, noise, continuum-subtracted, stellar model per bin).

4. **`04_fit_emission_lines_bins.py`** (bin-level, default) — Per-bin Gaussian fitting of all 12 emission lines simultaneously. Single shared velocity and velocity dispersion per bin; [OIII]4959/5007 and [NII]6548/6584 doublet amplitudes are tied at atomic ratios. MUSE instrumental broadening is accounted for. S/N ≥ 3 threshold for detection. Outputs `bin_kinematics.npz` + `kinematics_bins.fits` + bin-level FITS intensity maps in `bin_line_maps/`.

5. **`05_bpt_diagrams_bins.py`** (bin-level, default) — Computes N-BPT, S-BPT, and O-BPT line ratios from bin-level fluxes, classifies each bin using Kauffmann+03 / Kewley+01/06 demarcation lines, produces multi-page PDF with scatter (one point per bin) + spatial map panels in `outputs/bpt/`.

6. **`06_interactive_viz.py`** — Interactive matplotlib tool: Voronoi bin map with hover highlighting + click-to-inspect spectral fits per bin.

### Legacy spaxel-level scripts (--spaxel flag)

- **`03_subtract_continuum.py`** — Per-spaxel continuum subtraction with local rescaling factor.
- **`04_fit_emission_lines.py`** — Multiprocessed per-spaxel emission line fitting. Worker count controlled by `PIPE_N_WORKERS` env var.
- **`05_bpt_diagrams.py`** — Per-spaxel BPT classification.

### Config / shared modules

- **`scripts/common.py`** — Shared constants and helpers. Reads galaxy-specific config from env vars (`PIPE_*`). Contains the emission line list (Cresci 2017), doublet ratios, MUSE LSF function, `load_cube()`, and `measure_systemic_velocity()` (Hα cross-correlation). Key constants: `TARGET_SN_BIN=30.0`, `SN_DETECT=3.0`.
- **`scripts/00_common.py`** — Legacy single-galaxy version. Not used by the pipeline driver.
- **`scripts/bpt_demarcation_lines_v3.py`** — Standalone BPT demarcation line visualization (Cresci+2017 Fig. 5 style).

### Input data

- **`DWALIN_Sample/`** — 24 MUSE datacubes (`.fits`). Each cube has 3 extensions: PRIMARY (header), DATA (flux, shape nλ×ny×nx), STAT (variance).
- HEN_2-10 is excluded from the automated pipeline run.

### Output structure

```
outputs/
  <galaxy_name>/               Per-galaxy outputs
    cont_SN_map.fits            Continuum S/N map (per spaxel)
    voronoi_bin_map.fits        Bin assignments (-1 = excluded)
    voronoi_bins.npz            Bin geometry + stats
    ppxf_bin_fits.npz           Stellar & gas best-fit models per bin
    bin_spectra.npz             Bin-integrated spectra (original, noise, cont-sub, stellar)
    bin_kinematics.npz          Bin-level v, σ, line fluxes & errors
    kinematics_bins.fits        2D V and σ maps (bin-level)
    bin_line_maps/              Per-line FITS files at bin resolution (FLUX, FERR, SN, FLUX_SN3)
  bpt/
    <galaxy>_bpt.pdf            BPT diagnostic diagrams (one point per Voronoi bin)
```

### BPASS templates

BPASS v2.2.1 (binary SSP, IMF slope -1.35, 300 M☉ upper cutoff) pre-processed into `templates/bpass_processed/`:

- `bpass_templates_raw.npz` — 8 metallicities (z001–z020) × 51 ages (1 Myr–100 Gyr)
- Templates are log-rebinned and convolved to MUSE LSF at load time in stage 2

Set `PIPE_TEMPLATE_LIB=bpass` (default) or `PIPE_TEMPLATE_LIB=emiles` to switch.

## Data conventions

- MUSE cubes are stored as `data[λ, y, x]` (wavelength axis first; `NAXIS1 = λ` in FITS but numpy transposes this)
- Emission line wavelengths are rest-frame air wavelengths (Å) from Cresci et al. 2017
- Systemic velocity correction: observed λ = rest λ × (1 + Z_SYS) where Z_SYS = V_SYS / c
- V_SYS is auto-measured per galaxy via Hα+[NII] cross-correlation (`measure_systemic_velocity()`)
- Instrumental broadening: MUSE LSF FWHM(λ) = 5.835×10⁻⁸ λ² − 9.080×10⁻⁴ λ + 5.983 (Bacon+2017)

## Key findings

- Galaxies with many Voronoi bins (Haro11: 1,835, NGC_1487: 1,865) take ~4h for stage 2 — this is from high continuum S/N creating many 1-spaxel bins, not a bug
- Haro11_P1 shows strong shock/AGN signatures: 56% Seyfert in S-BPT, 60% in O-BPT
- Galaxies with weak Hβ (ESO321-14: 782 detections, VCC0170: 300) have sparse BPT diagrams — this is intrinsic, not a pipeline issue
- SDSSJ124615 has suspect v_sys (-23 km/s, cc=0.29) due to weak Hα — verify against literature
