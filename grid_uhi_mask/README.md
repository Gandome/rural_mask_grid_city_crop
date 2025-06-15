# City Climate Stats

This project computes climate statistics (mean, median, std, etc.) over urban areas using gridded climate data (NetCDF) and city boundary polygons.

## Structure
- `data/`: input datasets (NetCDF files, GeoPackage).
- `results/`: output CSV files with climate statistics.
- `plots/`: optional plots for each city and period.
- `masked_vals_*`: NetCDFs of masked data for each time period.
- `climate_city_stats.py`: core functions.
- `run_analysis.py`: script to execute the workflow.

## Requirements
See `requirements.txt`.

## Run
```bash
python run_analysis.py
```
