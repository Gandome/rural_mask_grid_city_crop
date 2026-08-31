# Changelog

## 2.0.1 — 2026-08-31

### Grid-alignment patch
- Increased the default PGD/tas coordinate-verification tolerance from `1e-5°` to `1e-4°`.
- The change addresses harmless lon/lat rounding differences observed in the public ALPX3 reproducibility test: approximately `±5e-5°`, corresponding to at most about 5.6 m, versus an approximately 2.5 km model grid.
- Grid verification remains mandatory. The tolerance does not trigger regridding, interpolation, or coordinate replacement.
- Added explicit `--coord-tolerance 1e-4` to the ALPX3 validation and production examples.
- Added regression tests confirming that rounding-level offsets are accepted by the v2.0.1 default and larger offsets are rejected.

### Packaging and metadata
- Updated package/software metadata and method-version attributes to `2.0.1`.
- Added v2.0.1 release and GitHub-publication notes.
- Removed cached pytest artifacts from the public ZIP.

## 2.0.0 — 2026-08-24

### UHI / MOD_Mask
- Replaced hierarchical urban-first classification with sea/water-first land masking.
- Added combined sea+water exclusion threshold.
- Added `rural_reference.py` and a reusable `RuralReference` data model.
- Rural-reference search is now static and can be precomputed once per sensitivity experiment.
- Search threshold is reduced only to the configured floor (default 50%); failure at the floor returns an invalid urban cell.
- Removed the historical fallback that could replace an all-NaN elevation difference with zero.
- Elevation-limited rural temperature now uses only rural cells passing the corresponding elevation threshold.
- Added time-vectorized UHI computation (`calculation.py`).
- Replaced nested time-step threading with process-level file parallelism.
- Added automatic Kelvin/Celsius handling and safer dimension standardization.
- Made lon/lat bounds optional rather than mandatory.
- Added 2-D diagnostics: accepted ratio, accepted threshold, radius, candidate/reference counts, search success, and reference-use frequency.
- Added method metadata documenting the failed-search rule and lapse-rate equation.

### City/rural extraction
- Added `standalone_plot_uhi_city_rural_overlay_RAW.py` and `_v2.py`.
- Added `exact` rural mode that reconstructs the actual MOD_Mask reference cells from the PGD file and UHI metadata.
- Retained `radius_box` as a compatibility fallback and labels it accordingly.
- Raw UHI remains unnormalized and missing city UHI values are not fabricated.
- Changed map rendering from `imshow(extent=...)` to native 2-D `pcolormesh` for curvilinear grids.
- Uses the actual GHSL polygon boundary for the plotted city outline.
- NetCDF city products now include `rural_reference_count` as well as masks.

### Packaging
- Updated both subpackage versions to 2.0.0.
- Removed standard-library modules from `requirements.txt`.
- Removed the accidental Vim swap file.
- Added synthetic regression tests for the adaptive rural search and lapse-rate correction.

### Public-release hardening
- Added CLI-based `--pgd`, `--tas`, and `--output` configuration so users do not need to edit source code.
- Added explicit support for single yearly tas files, directories, and glob patterns.
- Added preflight PGD/tas grid verification and safe contiguous subdomain matching from native 2-D lon/lat coordinates.
- Added `--validate-only` mode.
- Added the corrected ALPX3 author example using `PGD_ALPX3_selected_alpine_cities_buffer_1p0deg.nc` and the 2000 hourly AROME46t1 tas file.
- Added command-line configuration to the standalone city/rural overlay script; recommended `exact` mode is now the default.
- Added GitHub Actions, `CITATION.cff`, release notes, contributing guidance, and a conda environment file.
