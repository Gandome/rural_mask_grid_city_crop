# grid_uhi_mask — MOD_Mask/UHI Version 2.0.1

`grid_uhi_mask` computes an urban heat island field for every model urban grid cell by comparing its temperature with an adaptively selected rural reference mask.

## Version-2 algorithm

For each sensitivity experiment, static PGD masks are constructed once. Sea and inland-water cells are excluded first using the combined fraction. Urban and rural eligibility are then evaluated over land.

For each urban cell, the algorithm starts from an outer half-width `nbg` and excludes a central square of half-width `nO`. Candidate cells are non-sea/water cells in the remaining region. A rural candidate additionally satisfies the rural and urban fraction thresholds. The accepted ratio is

`100 * N_rural / N_total`.

The outer radius is expanded for `max_iterations`. If the requested ratio is not achieved, the ratio requirement is reduced by `ratio_step` down to `Min_Value_floor` (default 50%). If even the floor cannot be reached, the urban cell is invalid and UHI remains NaN.

The resulting reference indices are precomputed once and reused for every NetCDF file and every time step.

## Elevation-aware variants

For each requested limit `LR`, only reference cells satisfying

`abs(z_urban - z_rural) <= LR`

are retained. Their time-dependent mean temperature is adjusted to the urban elevation:

`T_r_adj = mean(T_r) - 0.0065 * mean(z_urban - z_rural)`.

`UHI_LR = T_urban - T_r_adj`.

## Main output variables

- `UHI_px`
- `UHI_LR100`, `UHI_LR200`, `UHI_LR300`, `UHI_LR500` (or configured limits)
- `rural_temperature_mean`
- `rural_temperature_LR*_mean`
- `Ratio_used`
- `Min_Value_used`
- `nbg`
- `n_total_reference`
- `n_rural_reference`
- `rural_search_success`
- `rural_reference_frequency`

## Grid compatibility

Before any expensive rural-reference search, PGD and `tas` native 2-D lon/lat
coordinates are verified. Version 2.0.1 uses a default tolerance of `1e-4` degree
to accommodate harmless coordinate rounding (for example, differences of order
`5e-5` degree) while still rejecting materially displaced grids. This check never
regrids or interpolates data. Override it explicitly with `--coord-tolerance` when
needed.

## Run

Use the public CLI; source-code editing is not required:

```bash
python -u grid_uhi_mask/scripts/run_UHI_process_parallel.py \
  --pgd /path/to/PGD.nc \
  --tas /path/to/tas.nc \
  --output /path/to/UHI_MOD_MASK_V2_0_1 \
  --coord-tolerance 1e-4 \
  --validate-only
```

After a successful preflight, remove `--validate-only` to run the calculation.
