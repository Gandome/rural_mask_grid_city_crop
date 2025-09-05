# City Climate Stats

This project computes climate statistics (mean, median, std, etc.) over urban areas using gridded climate data (NetCDF) and city boundary polygons.


## Extraction and Visualization of Cities from Gridded Climate Data

Our package, `clim_city_mask`, imports the necessary Python libraries for geospatial tasks, numerical analysis, and visualization. By default, the package uses the 2024 European urban boundaries GeoPackage. However, users can provide other datasets, like Copernicus Human Settlements. The package matches city boundary polygons with NetCDF climate data produced from the rural mask workflow. This process allows users to choose cities of interest, such as Paris or Dijon, via unique identifiers in the GeoPackage. The climate grid is then converted into an array of Shapely polygons, where each cell is defined by connecting its four corner points using the longitude and latitude arrays.

For the selected city, the urban boundary polygon is checked and simplified if necessary, keeping the largest component if MultiPolygons are present. A fractional mask, `frac_mask`, is then calculated. This mask measures the overlap of each grid cell with the urban boundary, assigning values that range from 0 (no overlap) to 1 (full coverage). This mask is used to crop the relevant subset of climate data. This approach ensures that computations only involve cells that contribute to the urban area. By using fractional coverage instead of simple yes/no inclusion, the method produces more precise and representative urban climate statistics.

The package calculates key statistical metrics, such as mean, median, standard deviation, minimum, and maximum, for the cropped urban area at user-defined time resolutions (daily, monthly, or seasonal). Users can generate visualizations as color-coded maps that overlay climate variables, like temperature, precipitation, and humidity, onto city outlines. This setup helps identify spatial patterns and climate anomalies within urban areas. The workflow supports batch processing, which allows simultaneous analysis of multiple cities and time periods. This feature makes it ideal for large-scale comparative studies across different regions and timelines.

For applications at the country level, the Countries_based_cities_data_extraction.py module automates the extraction and analysis of gridded climate data for all recorded cities within a nation. Using high-resolution city boundary GeoPackages, this module accurately masks NetCDF climate datasets to include only urban-intersecting grid cells. It can also export city- and country-level statistical summaries in CSV format while maintaining the masked NetCDF data for each city. This integrated system offers a reproducible, scalable, and dependable framework for studying urban climate dynamics at both local and national levels. It supports research and policy applications effectively.


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
