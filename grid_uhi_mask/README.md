# grid_uhi_mask — MOD_Mask/UHI Version 2

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

## Run

Edit `scripts/run_UHI_process_parallel.py`, then:

```bash
python scripts/run_UHI_process_parallel.py
```
