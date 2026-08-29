# MOD_Mask v2 — rural mask and urban heat island workflow

[![tests](https://github.com/Gandome/rural_mask_grid_city_crop/actions/workflows/tests.yml/badge.svg)](https://github.com/Gandome/rural_mask_grid_city_crop/actions/workflows/tests.yml)

`rural_mask_grid_city_crop` provides the **MOD_Mask** adaptive rural-reference method for gridded Urban Heat Island (UHI) calculation together with GHSL-based city extraction and publication-quality city/rural maps.

**Current public version: 2.0.0**

## What is new in v2

- Sea/water is excluded before urban/rural classification.
- The rural-reference geometry is computed once per urban cell and reused for every time step and input file.
- The requested rural availability ratio (default 70%) is reduced by 5 percentage points only down to a hard 50% floor.
- If the 50% floor is still not reached after the maximum search extent, the urban cell is invalid and its UHI remains `NaN`.
- Elevation-filtered UHI uses only rural cells satisfying `|z_urban-z_rural| <= LR`, with standard limits of 100, 200, 300 and 500 m.
- Static diagnostics are written to NetCDF: `Ratio_used`, `Min_Value_used`, `nbg`, `n_total_reference`, `n_rural_reference`, `rural_search_success`, and `rural_reference_frequency`.
- The runner now accepts explicit files, directories or glob patterns and validates PGD/tas grid compatibility before computation.
- A rectangular PGD subdomain can be matched safely to a larger tas grid when 2-D lon/lat coordinates prove the alignment.
- The standalone city plot can reconstruct the **exact MOD_Mask rural footprint**, not merely an expanded bounding box.

## Repository layout

```text
rural_mask_grid_city_crop/
├── grid_uhi_mask/          # MOD_Mask + UHI calculation
├── clim_city_mask/         # GHSL city extraction and plotting
├── examples/               # reproducible command-line examples
├── tests/                  # regression tests
├── .github/workflows/      # continuous integration
├── CITATION.cff
├── RELEASE_NOTES_v2.0.0.md
└── LICENSE
```

## Installation

### Conda

```bash
git clone https://github.com/Gandome/rural_mask_grid_city_crop.git
cd rural_mask_grid_city_crop
conda env create -f environment.yml
conda activate mod-mask-v2
pip install -e ./grid_uhi_mask
pip install -e ./clim_city_mask
```

### Existing Python environment

```bash
pip install -r grid_uhi_mask/requirements.txt
pip install -r clim_city_mask/requirements.txt
pip install -e ./grid_uhi_mask
pip install -e ./clim_city_mask
```

Python 3.10+ is recommended.

## Quick preflight test

Before launching an expensive UHI computation:

```bash
python grid_uhi_mask/scripts/run_UHI_process_parallel.py \
  --pgd /path/to/PGD.nc \
  --tas /path/to/tas.nc \
  --output ./UHI_MOD_MASK_V2 \
  --validate-only
```

This verifies required PGD fields, input discovery, horizontal grid compatibility and urban/rural mask construction.

## Run MOD_Mask/UHI

```bash
python -u grid_uhi_mask/scripts/run_UHI_process_parallel.py \
  --pgd /path/to/PGD.nc \
  --tas /path/to/tas_2000.nc \
  --output /path/to/UHI_MOD_MASK_V2 \
  --min-values 70 \
  --sea-water-thresholds 0.30 \
  --urban-thresholds 0.20 \
  --rural-thresholds 0.60 \
  --nO 2 \
  --initial-nbg 4 \
  --max-iterations 26 \
  --min-ratio-floor 50 \
  --ratio-step 5 \
  --height-limits 100,200,300,500 \
  --lapse-rate 0.0065 \
  --nproc 1
```

`--tas` may receive one or more explicit NetCDF files, directories, or glob patterns. Test first with `--nproc 1`; increase file-level parallelism only after a successful real-data run.

### Author ALPX3 2000 example

The repository includes `examples/run_alpx3_2000.sh` using:

```text
PGD = /archive/globc/quenum/Results/Data_UHI_Paper/PGD_ALPX3_selected_alpine_cities_buffer_1p0deg.nc

tas = /archive/globc/quenum/model_output/AROME/ALPX3/Evaluation/1hr/tas/
      tas_ALPX-3_ERA5_evaluation_r1i1p1f1_CNRM-MF_CNRM-AROME46t1_v1-r1_1hr_200001010100-200101010000.nc
```

These machine-specific paths are provided only to make the author's experiment reproducible; public users should replace them with their own data locations.

Run only the preflight:

```bash
bash examples/validate_alpx3_2000.sh
```

Run the calculation:

```bash
bash examples/run_alpx3_2000.sh
```

## UHI definitions

For urban grid cell `u`, a candidate rural grid cell `r` must satisfy

```text
FRAC_NATURE(r) >= rural_threshold
FRAC_TOWN(r)   <= urban_threshold
FRAC_SEA(r) + FRAC_WATER(r) <= sea_water_threshold
```

and must lie outside the central exclusion square of half-width `nO`.

The rural availability ratio is

```text
Ratio = 100 × N_rural / N_total
```

where `N_total` contains eligible non-sea/water cells in the clipped search window outside the inner exclusion region.

The basic UHI is

```text
UHI_px(t) = T_urban(t) - mean(T_rural(t))
```

For elevation threshold `LR`, only rural cells with

```text
|z_urban - z_rural| <= LR
```

are retained. Their rural temperature is adjusted to urban elevation using

```text
T_rural,LR,adj(t)
    = mean(T_rural,LR(t))
      - Gamma × mean(z_urban - z_rural)

Gamma = 0.0065 K m-1
```

and

```text
UHI_LR(t) = T_urban(t) - T_rural,LR,adj(t)
```

## Main NetCDF output fields

```text
UHI_px
UHI_LR100
UHI_LR200
UHI_LR300
UHI_LR500
rural_temperature_mean
rural_temperature_LR100_mean
rural_temperature_LR200_mean
rural_temperature_LR300_mean
rural_temperature_LR500_mean
Ratio_used
Min_Value_used
nbg
n_total_reference
n_rural_reference
rural_search_success
rural_reference_frequency
```

## City/rural UHI overlay

The recommended mode is `exact`:

```bash
python clim_city_mask/scripts/standalone_plot_uhi_city_rural_overlay_RAW_v2.py \
  --uhi-dir /path/to/UHI/Min70_sea0p30_urb0p20_rur0p60 \
  --pgd /path/to/PGD.nc \
  --city-gpkg clim_city_mask/data/GHS_UCDB_REGION_EUROPE_R2024A.gpkg \
  --cities Grenoble Chambery Geneva \
  --rural-mode exact \
  --output ./output_city_uhi_figures_RAW_v2
```

Raw UHI values are preserved. The display color limits do not normalize or alter NetCDF values.

## Tests

```bash
python -m compileall -q grid_uhi_mask clim_city_mask
python -m pytest -q
```

GitHub Actions runs the core regression suite automatically on pushes and pull requests.

## Citation

Please use the metadata in `CITATION.cff` to cite the software. For scientific publications, also cite the associated MOD_Mask methodological paper once its final bibliographic information is available.

## License

MIT License. See `LICENSE`.
