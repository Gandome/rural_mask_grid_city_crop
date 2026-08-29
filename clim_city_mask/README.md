# clim_city_mask — Version 2

This package extracts and summarizes gridded climate/UHI information over GHSL urban-centre polygons and provides standalone city/rural visualizations.

## Recommended Version-2 UHI overlay

Use:

```bash
python scripts/standalone_plot_uhi_city_rural_overlay_RAW.py
```

The script preserves raw `UHI_px` values and supports two rural-reference modes:

- `RURAL_MODE = "exact"` — recommended when the matching PGD is available. Reconstructs the actual MOD_Mask rural reference cells for model urban cells within each city using the same PGD fields and the method parameters stored in the UHI NetCDF.
- `RURAL_MODE = "radius_box"` — compatibility fallback. Shows the old bounding-box region inferred from `nbg`; it is not the exact rural mask.

Because AROME/ALPX3 uses a curvilinear projected grid, the v2 map uses 2-D `pcolormesh(lon, lat, ...)` rather than `imshow` with a simple rectangular extent.

Per-city NetCDF outputs include:

- `city_polygon_mask`
- `model_urban_in_city`
- `<UHI_VAR>_raw_city_overlay`
- `raw_uhi`
- `rural_reference_mask`
- `rural_reference_count`

The bundled GHSL GeoPackage remains in `data/`.
