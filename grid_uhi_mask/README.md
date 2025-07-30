# City Climate Stats

This project computes climate statistics (mean, median, std, etc.) over urban areas using gridded climate data (NetCDF) and city boundary polygons.

## Structure
- `data/`: input datasets (NetCDF files, GeoPackage).
- `results/`: output CSV files with climate statistics.
- `plots/`: optional plots for each city and period.
- `grid_uhi_mask/`: folder containing all the needed process for the reference rural areas identification.
- `grid_uhi_mask/spatial_UHI_mask/`: subset of the functions. They provide compuation of the UHI over each grid as well as information about the extend of the rural areas (`nbg`)
- `grid_uhi_mask/spatial_UHI_mask/urban_mask.py`: extract urban mask
- `grid_uhi_mask/spatial_UHI_mask/parallel_process.py`: Allow a parrallel running
- `grid_uhi_mask/spatial_UHI_mask/calculation_process.py`: Calculate the UHI
- `masked_vals_*`: NetCDFs of masked data for each time period.
- `masked_vals_*`: NetCDFs of masked data for each time period.



- `climate_city_stats.py`: core functions.
- `run_analysis.py`: script to execute the workflow.

## Requirements
See `requirements.txt`.

## Run
```bash
python run_analysis.py
```

```bash
python run_UHI_process_parallel.py
```
