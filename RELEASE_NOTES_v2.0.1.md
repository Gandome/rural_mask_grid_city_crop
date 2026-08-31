# MOD_Mask v2.0.1 release notes

Version 2.0.1 is a small compatibility and reproducibility patch to v2.0.0. The
scientific MOD_Mask rural-reference and UHI algorithms are unchanged.

## What changed

- The default PGD/tas lon/lat verification tolerance is now `1e-4°` instead of `1e-5°`.
- The public ALPX3 test revealed symmetric coordinate-rounding differences of about `±5e-5°` despite identical `(667, 847)` grid shapes. The measured maximum displacement was approximately 5.6 m in latitude and 3.9 m in longitude, negligible relative to the approximately 2.5 km AROME grid.
- Grid-coordinate verification is still mandatory and still fails when the configured tolerance is exceeded. No regridding, interpolation, or coordinate replacement is introduced.
- For substantially finer or higher-precision grids, users should explicitly choose a smaller tolerance after inspecting native lon/lat differences.
- `examples/validate_alpx3_2000.sh` and `examples/run_alpx3_2000.sh` now pass `--coord-tolerance 1e-4` explicitly.
- Regression tests cover both acceptance of rounding-level coordinate differences and rejection of larger mismatches.
- Software/package metadata and NetCDF `method_version` metadata are updated to `2.0.1`.

## Recommended validation

```bash
python -m pytest -v
bash examples/validate_alpx3_2000.sh
```

The validation step should report `Grid compatibility: PASSED` for the documented
ALPX3 PGD/tas pair while continuing to protect against genuine grid misalignment.

## Upgrade from v2.0.0

No data-format migration is required. Existing v2.0.0 commands remain valid. If a
workflow intentionally requires a stricter tolerance, pass it explicitly, e.g.
`--coord-tolerance 1e-5`.
