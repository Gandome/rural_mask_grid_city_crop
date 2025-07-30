# City Climate Stats

This project computes climate statistics (mean, median, std, etc.) over urban areas using gridded climate data (NetCDF) and city boundary polygons.

## Structure
- `data/`: input datasets (NetCDF files, GeoPackage).
- `results/`: output CSV files with climate statistics.
- `plots/`: optional plots for each city and period.
- `clim_city_mask/`: folder containing all the needed process for the reference rural areas identification.
- `clim_city_mask/scripts/`: subset of the functions. They provide compuation of the studied variables over country/countries and city/cities .

- `climate_city_stats.py`: core functions.
- `run_analysis.py`: script to execute the workflow.

## Requirements
See `requirements.txt`.

## Run
```bash
python run_analysis.py
```

```bash
python Countries_based_cities_data_extraction.py
```

```bash
python Map_City_Rural_aera.py
```

```bash
python population_and_climate_data.py
```
