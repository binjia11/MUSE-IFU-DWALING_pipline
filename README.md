# Low-Metallicity Shocks in the LMC — MUSE IFU Pipeline

Pipeline for analyzing ionized gas shocks in low-metallicity dwarf galaxies using MUSE integral-field spectroscopy. The code performs Voronoi binning (S/N ≥ 30), pPXF stellar population fitting (BPASS v2.2.1 binary SSP templates), **bin-level** emission line fitting, and **bin-level** BPT diagnostic classification for 23 galaxies from the DWALIN survey and additional low-metallicity targets.

## Requirements

- Python 3.9+ with a working conda environment
- Packages: `numpy`, `scipy`, `astropy`, `matplotlib`, `ppxf`, `vorbin`

The pipeline was developed with conda environment `uclchem_3.4`:

```bash
conda create -n uclchem_3.4 python=3.9 numpy scipy astropy matplotlib
conda activate uclchem_3.4
pip install ppxf vorbin
```

## Stellar templates

**BPASS v2.2.1** (default): Binary SSP models, IMF slope −1.35, 300 M☉ upper cutoff. 8 metallicities (Z = 0.001–0.020) × 51 ages (1 Myr–100 Gyr). Pre-processed templates are included in `templates/bpass_processed/`.

To use EMILES instead, set the environment variable:

```bash
export PIPE_TEMPLATE_LIB=emiles
```

If using EMILES, download the templates to the pPXF `sps_models` directory:

```bash
curl -L -o "$(python -c 'import ppxf; import os; print(os.path.join(os.path.dirname(ppxf.__file__), "sps_models", "spectra_emiles_9.0.npz"))')" \
  https://raw.githubusercontent.com/micappe/ppxf_data/main/spectra_emiles_9.0.npz
```

## Data

MUSE datacubes should be placed in `DWALIN_Sample/` at the project root. Each `.fits` file must be a MUSE cube with three extensions: PRIMARY (header), DATA (flux, shape nλ×ny×nx), and STAT (variance).

The 23-galaxy sample includes: AGC7983, ESO154-023, ESO320-14, ESO321-14, ESO379-7, ESO379-G024, ESO489-G56, Haro11_P1, IIZW40, J0908+0517, J1151-0222, NGC1705, NGC2915, NGC3125, NGC_1487, NGC_5253, NGC_625, SDSS115237-022806, SDSSJ112711.0+084353, SDSSJ124615.2+101220, Tol_1924-416, Tol_65, VCC0170.

Data can be obtained from the [ESO Science Archive](https://archive.eso.org/).

## Running

**Full pipeline (all galaxies, auto-skips completed):**

```bash
# Bin-level pipeline (default) — emission lines + BPT at Voronoi-bin resolution
python3 scripts/run_pipeline.py

# Parallel (N galaxies concurrently, 1 dedicated core each)
python3 scripts/run_pipeline.py --parallel        # all available cores
python3 scripts/run_pipeline.py --parallel 4      # 4 galaxies at a time

# Legacy per-spaxel pipeline
python3 scripts/run_pipeline.py --spaxel
```

The driver auto-discovers all `.fits` files, measures systemic velocity via Hα cross-correlation, and skips galaxies with existing BPT PDFs. No timeouts — runs each stage to completion.

**Individual stage for a single galaxy:**

```bash
export PIPE_GALAXY=ESO154-023
export PIPE_CUBE_PATH="$(pwd)/DWALIN_Sample/ESO154-023.fits"
export PIPE_OUT_DIR="$(pwd)/outputs/ESO154-023"
export PIPE_V_SYS=581
export PIPE_TEMPLATE_LIB=bpass
python3 scripts/04_fit_emission_lines_bins.py
```

**Interactive visualization (post-pipeline):**

```bash
python3 scripts/06_interactive_viz.py ESO154-023
```

Hover over the Voronoi bin map to highlight bins; click to view the full spectrum (original, continuum-subtracted, stellar model) and per-line zoom-in panels with Gaussian emission-line fits.

## Pipeline stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `01_voronoi_binning.py` | Continuum S/N → Voronoi binning to S/N ≥ 30 |
| 2 | `02_ppxf_fit_bins.py` | pPXF stellar continuum fit per bin (BPASS v2.2.1 + gas templates) |
| 3 | `03_subtract_continuum_bins.py` | **Bin-level** continuum subtraction + local polynomial refinement |
| 4 | `04_fit_emission_lines_bins.py` | **Bin-level** simultaneous 12-line Gaussian fitting |
| 5 | `05_bpt_diagrams_bins.py` | **Bin-level** N-BPT, S-BPT, O-BPT classification and PDF output |
| — | `06_interactive_viz.py` | Interactive bin-map explorer with spectral zoom-ins |

## Key design choices

- **Bin-level analysis**: Emission lines and BPT classification are performed on Voronoi-bin integrated spectra, not individual spaxels. This gives higher S/N per measurement and faster processing.
- **Voronoi S/N ≥ 30**: Continuum S/N measured in 5300–5530 Å rest-frame. Lower threshold produces more bins for better spatial sampling.
- **Systemic velocity**: Auto-measured per galaxy via Hα+[NII] cross-correlation.
- **Line fitting**: All 12 lines fitted simultaneously with a single shared v, σ per bin. [OIII] and [NII] doublet ratios tied at atomic values. MUSE instrumental broadening (Bacon+2017) accounted for.
- **Detection threshold**: S/N ≥ 3 for emission line detection.
- **Parallel mode**: Each galaxy gets 1 dedicated core (`OMP_NUM_THREADS=1`). Fast galaxies finish early, cores freed for remaining ones.
- **BPASS v2.2.1** stellar templates (binary SSP, 1 Myr–100 Gyr, 8 metallicities).

## Output structure

```
outputs/
  <galaxy>/
    cont_SN_map.fits           Continuum S/N per spaxel
    voronoi_bin_map.fits        Bin assignments (-1 = excluded)
    voronoi_bins.npz            Bin geometry & stats
    ppxf_bin_fits.npz           Stellar & gas best-fit models per bin
    bin_spectra.npz             Bin-integrated spectra (original, noise, cont-sub)
    bin_kinematics.npz          Bin-level v, σ, line fluxes & errors
    kinematics_bins.fits        2D velocity & dispersion maps (bin-level)
    bin_line_maps/              Per-line FITS maps (FLUX, FERR, SN, FLUX_SN3 at bin resolution)
  bpt/
    <galaxy>_bpt.pdf            BPT diagnostic diagrams (one point per bin)
```

## Science highlights

- **Haro11_P1** (1,835 bins): 56% Seyfert in S-BPT, 60% in O-BPT — strong shock/AGN signatures
- **NGC_1487** (1,865 bins, 94K spaxels): Largest galaxy in the sample
- Galaxies with many Voronoi bins (1,000+) take hours for stage 2 — this is from high continuum S/N creating many single-spaxel bins, not a code issue
- Several galaxies (ESO321-14, VCC0170) have sparse BPT diagrams due to intrinsically weak Hβ

## References

- Cresci et al. 2017 — emission line list and BPT methodology
- Bacon et al. 2017 — MUSE LSF parameterization
- Kewley et al. 2001, 2006; Kauffmann et al. 2003 — BPT demarcation lines
- Stanway & Eldridge 2018 — BPASS v2.2.1 stellar population models
- Relevant papers in `papers/`
