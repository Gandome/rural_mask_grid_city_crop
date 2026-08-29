# MOD_Mask v2.0.0 release notes

Version 2 is the revised public implementation of the MOD_Mask rural-reference/UHI workflow.

## Scientific changes

- Sea/water cells are excluded before urban/rural classification.
- Rural-reference geometry is computed once per urban cell and reused over time.
- Rural availability starts at the requested threshold and decreases in 5 percentage-point steps to a hard 50% floor.
- If the 50% floor is not satisfied after the maximum search extent, the urban cell is invalid and UHI remains `NaN`.
- Elevation-limited rural means first filter cells by `|z_urban-z_rural|`, then compute the rural temperature mean and lapse-rate adjustment.
- Standard elevation limits are 100, 200, 300 and 500 m.
- Static search diagnostics are included in every output NetCDF.

## Public-software changes

- New CLI accepts explicit `--pgd`, `--tas`, and `--output` paths.
- `--tas` accepts files, directories, and glob patterns.
- PGD/tas compatibility is checked before the expensive reference search.
- Rectangular PGD subsets can be matched to a larger tas grid when 2-D lon/lat coordinates prove the alignment.
- `--validate-only` provides a fast preflight check.
- File-level multiprocessing is retained but can be tested safely with `--nproc 1`.
- Standalone GHSL city/rural plotting supports CLI arguments and defaults to scientifically recommended `exact` rural-reference reconstruction.
- GitHub Actions tests and software citation metadata are included.
