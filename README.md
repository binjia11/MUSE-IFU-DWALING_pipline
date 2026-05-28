# Low-Metallicity Shocks in the LMC — MUSE IFU Pipeline

Pipeline for analyzing ionized gas shocks in low-metallicity dwarf galaxies using MUSE integral-field spectroscopy. The code performs Voronoi binning, pPXF stellar population fitting, emission line fitting, and BPT diagnostic classification for a sample of galaxies from the DWALIN survey.

## Requirements

- Python 3.9+ with a working conda environment
- Packages: `numpy`, `scipy`, `astropy`, `matplotlib`, `ppxf`, `vorbin`

The pipeline was developed with conda environment `uclchem_3.4`:

```bash
conda create -n uclchem_3.4 python=3.9 numpy scipy astropy matplotlib
conda activate uclchem_3.4
pip install ppxf vorbin
```

## EMILES templates

pPXF requires the EMILES stellar population templates. Download to the pPXF `sps_models` directory:

```bash
curl -L -o "$(python -c 'import ppxf; import os; print(os.path.join(os.path.dirname(ppxf.__file__), "sps_models", "spectra_emiles_9.0.npz"))')" \
  https://raw.githubusercontent.com/micappe/ppxf_data/main/spectra_emiles_9.0.npz
```

## Data

MUSE datacubes for the DWALIN sample galaxies should be placed in a `DWALIN_SAMPLE/` directory at the project root. Each `.fits` file must be a MUSE cube with three extensions: PRIMARY (header), DATA (flux, shape nλ×ny×nx), and STAT (variance).

The sample includes: ESO154-023, ESO320-14, ESO321-14, ESO379-7, ESO379-G024, ESO489-G56, HEN_2-10, Haro11_P1, IIZW40, SDSSJ112711.0+084353, VCC0170.

Data can be obtained from the [ESO Science Archive](https://archive.eso.org/).

## Running

**Full pipeline for all sample galaxies:**

```bash
python3 scripts/run_pipeline.py
```

This runs stages 01–05 sequentially for each galaxy, writing outputs to `outputs/<galaxy>/` and BPT PDFs to `outputs/bpt/`.

**Individual stage for a single galaxy:**

```bash
export PIPE_GALAXY=ESO154-023
export PIPE_CUBE_PATH="$(pwd)/DWALIN_SAMPLE/ESO154-023.fits"
export PIPE_OUT_DIR="$(pwd)/outputs/ESO154-023"
export PIPE_V_SYS=0.0
python3 scripts/02_ppxf_fit_bins.py
```

## Pipeline stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `01_voronoi_binning.py` | Continuum S/N → Voronoi binning to S/N ≥ 50 |
| 2 | `02_ppxf_fit_bins.py` | pPXF stellar continuum fit per bin (EMILES + gas templates) |
| 3 | `03_subtract_continuum.py` | Subtract stellar continuum from each spaxel |
| 4 | `04_fit_emission_lines.py` | Multiprocessed Gaussian emission line fitting |
| 5 | `05_bpt_diagrams.py` | N-BPT, S-BPT, O-BPT classification and PDF output |

## Output structure

```
outputs/
  <galaxy>/
    cont_SN_map.fits
    voronoi_bin_map.fits
    voronoi_bins.npz
    ppxf_bin_fits.npz
    <galaxy>_cont_sub.fits
    kinematics.fits
    line_maps/
      Halpha.fits, Hbeta.fits, OIII5007.fits, NII6584.fits, ...
  bpt/
    <galaxy>_bpt.pdf
```

## References

- Cresci et al. 2017 — emission line list and BPT methodology
- Bacon et al. 2017 — MUSE LSF parameterization
- Kewley et al. 2001, 2006; Kauffmann et al. 2003 — BPT demarcation lines
- Relevant papers in `papers/`
