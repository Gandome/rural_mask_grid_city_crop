# Contributing

Contributions that improve portability, testing, documentation, computational efficiency, or scientific diagnostics are welcome.

1. Fork the repository and create a feature branch.
2. Keep scientific changes separate from purely cosmetic changes where possible.
3. Add or update tests for changes to rural-reference selection, grid alignment, lapse-rate correction, or NetCDF diagnostics.
4. Run:

```bash
python -m compileall -q grid_uhi_mask clim_city_mask
python -m pytest -q
```

5. Open a pull request describing both the software change and any scientific consequence.

Please do not commit model archives, credentials, machine-specific environments, or unpublished restricted datasets.
