# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MUSE IFU pipeline for studying low-metallicity galaxies with ionized gas shocks in the LMC/DWALIN sample. Performs Voronoi binning, pPXF stellar continuum fitting, emission line fitting, and BPT classification of ionization mechanisms (SF vs AGN vs shocks).

## Conda environment

```
/opt/anaconda3/envs/uclchem_3.4/bin/python3   (env name: uclchem_3.4)
```

Key packages: `astropy`, `numpy`, `scipy`, `matplotlib`, `ppxf`, `vorbin`.

## Running the pipeline

**Full pipeline for all sample galaxies:**

```bash
python3 scripts/run_pipeline.py
```

This driver iterates over MUSE cubes in `DWALIN_Sample/`, running stages 01–05 sequentially for each galaxy. Per-galaxy parameters are passed via environment variables (`PIPE_GALAXY`, `PIPE_CUBE_PATH`, `PIPE_OUT_DIR`, `PIPE_V_SYS`). Each stage has a 4-hour timeout.

**Individual stage for one galaxy** (from the project root):

```bash
export PIPE_GALAXY=ESO154-023
export PIPE_CUBE_PATH="/Users/binjia/Desktop/low-metallicity_shocks_LMC/DWALIN_Sample/ESO154-023.fits"
export PIPE_OUT_DIR="/Users/binjia/Desktop/low-metallicity_shocks_LMC/outputs/ESO154-023"
export PIPE_V_SYS=0.0
python3 scripts/02_ppxf_fit_bins.py
```

## Architecture

The pipeline has 5 sequential stages, each a standalone script, communicating through files on disk:

1. **`01_voronoi_binning.py`** — Computes continuum S/N per spaxel (5300–5530 Å rest-frame), then uses `vorbin` to adaptively bin spaxels to S/N ≥ 50. Outputs `voronoi_bin_map.fits` + `voronoi_bins.npz`.

2. **`02_ppxf_fit_bins.py`** — For each Voronoi bin: sums spaxel spectra, log-rebins, runs pPXF with EMILES SSP templates + Gaussian gas templates (3 kinematic components: stars, Balmer, forbidden). Fits over 4760–7400 Å observed-frame. Stores stellar continuum models and gas models per bin in `ppxf_bin_fits.npz`.

3. **`03_subtract_continuum.py`** — Rescales the per-bin stellar model to match each spaxel's local continuum median, subtracts it. Outside the MILES coverage (e.g. around [SIII]9069), applies local 3rd-order polynomial fits around each emission line (±60 Å window, masking line core ±12 Å). Outputs `<galaxy>_cont_sub.fits` (DATA + STAT + VALID extensions).

4. **`04_fit_emission_lines.py`** — Multiprocessed per-spaxel Gaussian fitting of all emission lines simultaneously. Single shared velocity and velocity dispersion per spaxel; [OIII]4959/5007 and [NII]6548/6584 doublet amplitudes are tied at atomic ratios. MUSE instrumental broadening is accounted for. S/N ≥ 3 threshold for detection. Outputs `kinematics.fits` and per-line FITS maps in `line_maps/`.

5. **`05_bpt_diagrams.py`** — Computes N-BPT, S-BPT, and O-BPT line ratios, classifies each spaxel using Kauffmann+03 / Kewley+01/06 demarcation lines, produces multi-page PDF with scatter + spatial map panels in `outputs/bpt/`.

### Config / shared modules

- **`scripts/common.py`** — Shared constants and helpers for the DWALIN sample pipeline. Reads galaxy-specific config from env vars (`PIPE_*`). Contains the emission line list (Cresci 2017), doublet ratios, MUSE LSF function, and `load_cube()`.
- **`scripts/00_common.py`** — Legacy single-galaxy version hardcoded for He 2-10 (V_SYS=873 km/s). Not used by the pipeline driver.
- **`scripts/bpt_demarcation_lines_v3.py`** — Standalone reference script for BPT demarcation line visualization (Cresci+2017 Fig. 5 style). Used as visual reference for stage 5.

### Input data

- **`DWALIN_Sample/`** — MUSE datacubes (`.fits`) for 11 galaxies. Each cube has 3 extensions: PRIMARY (header), DATA (flux, shape nλ×ny×nx), STAT (variance). All sample galaxies have RADVEL=0 in their headers (V_SYS=0).
- HEN_2-10 is excluded from the automated pipeline run (already processed separately).

### Output structure

```
outputs/
  <galaxy_name>/               Per-galaxy outputs
    cont_SN_map.fits            Continuum S/N map
    voronoi_bin_map.fits        Bin assignments (-1 = excluded)
    voronoi_bins.npz            Bin geometry + stats
    ppxf_bin_fits.npz           Stellar & gas best-fit models per bin
    <galaxy>_cont_sub.fits      Continuum-subtracted cube
    kinematics.fits             V and σ maps
    line_maps/                  Per-line FITS files (FLUX, FERR, SN, FLUX_SN3 extensions)
  bpt/
    <galaxy>_bpt.pdf            BPT diagnostic diagrams
```

### EMILES templates

pPXF requires `spectra_emiles_9.0.npz` in the pPXF sps_models directory. Download:

```bash
curl -L -o /opt/anaconda3/envs/uclchem_3.4/lib/python3.9/site-packages/ppxf/sps_models/spectra_emiles_9.0.npz \
  https://raw.githubusercontent.com/micappe/ppxf_data/main/spectra_emiles_9.0.npz
```

## Data conventions

- MUSE cubes are stored as `data[λ, y, x]` (wavelength axis first; `NAXIS1 = λ` in FITS but numpy transposes this)
- Emission line wavelengths are rest-frame air wavelengths (Å) from Cresci et al. 2017
- Systemic velocity correction: observed λ = rest λ × (1 + Z_SYS) where Z_SYS = V_SYS / c
- Instrumental broadening: MUSE LSF FWHM(λ) = 5.835×10⁻⁸ λ² − 9.080×10⁻⁴ λ + 5.983 (Bacon+2017)
