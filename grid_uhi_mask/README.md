# City Climate Stats

This project computes climate statistics (mean, median, std, etc.) over urban areas using gridded climate data (NetCDF) and city boundary polygons.

## Land Cover Classification

Land cover classification of grid cells uses the function `classify_grid_points`. The source code for this function and the entire data processing workflow is available at [this repository](https://github.com/Gandome/rural_mask_grid_city_crop/tree/main/grid_uhi_mask). This repository offers tools to merge land cover information with gridded climate datasets, crop data within urban boundaries, and create spatial masks to identify rural areas.  

The function `classify_grid_points` works with fractional land cover data to assign each grid cell to one of several categories: `urban`, `rural`, `water`, or `sea`. The classification uses threshold values that define the minimum fraction of a land cover type needed for a grid cell to be placed in that category. Input arguments include the fractional coverage of different land cover types:

- `town_fract` – urban areas  
- `nature_fract` – natural/rural areas  
- `sea_fract` – oceanic areas  
- `water_fract` – lakes and rivers  

Two threshold parameters, `urban_threshold` and `rural_threshold`, help determine the dominant type of land cover.  

The classification is done hierarchically using `xarray` conditional operations. First, cells are classified as urban if their `town_fract` exceeds `urban_threshold`. If this condition is not met, the rural fraction (`nature_fract`) is checked against `rural_threshold`. Cells that pass this test are classified as rural. The remaining cells are then examined for water or sea dominance, based on their fractional coverage.  

Currently, grid cells that do not meet any of these criteria remain unclassified and default to `NaN`. However, the code has a placeholder for a future option to assign such cells to an explicit `"others"` category.  

The output of `classify_grid_points` is an `xarray.DataArray` with string labels that show each grid cell’s classification. These labels (`"urban"`, `"rural"`, `"water"`, `"sea"`) are used for later analyses, allowing consistent masking, aggregation, and comparison of gridded climate variables across various land-use types.


## Definition of the Rural Reference Mask
In addition to the `classify_grid_points` function (see [Land Cover Classification](#land-cover-classification)), the function `extract_filtered_submatrix` ([source code](https://github.com/Gandome/rural_mask_grid_city_crop/tree/main/grid_uhi_mask)) has been developed to extract a neighborhood of grid cells around a specified urban point and identify suitable rural reference areas. This function is part of the `utils` module and is particularly important in urban climate studies, where selecting rural cells minimally influenced by urbanization is critical for comparative analysis.  
### Function Inputs
`extract_filtered_submatrix` requires multiple input parameters:
- `matrix` – primary data matrix (high-resolution climate simulations)  
- `matrix2` – secondary data matrix (can include other climate variables or land cover classifications)  
- `urban_point = (x, y)` – coordinates of the target urban grid cell  
- `nbg` – radius of the surrounding area to include in the submatrix (total size = `2*nbg + 1`)  
- `nO` – size of an inner exclusion zone around the urban cell to avoid nearby cells influenced by urban effects  
- `rural_threshold` – minimum fraction of rural land cover for a valid rural candidate  
- `urban_threshold` – maximum fraction of urban land cover for a valid rural candidate  
### Workflow Overview
1. **Submatrix Extraction:**  
   The function calculates the boundaries of the submatrix using the neighborhood size, ensuring the submatrix stays within the global matrix dimensions. Square submatrices are then extracted from the four input matrices: `matrix`, `matrix2`, `rural_frac`, and `urban_frac`.  

2. **Masking:**  
   Two boolean masks are created:  
   - **Inner exclusion mask:** Excludes the central urban region (`nO`) from consideration.  
   - **Boundary mask:** Ensures only cells safely within the global matrix and outside the urban center are considered.  
   These masks are combined into a `total_mask` of candidate cells.  
3. **Conditioning:**  
   A conditioning mask identifies cells satisfying all of the following:  
   - Rural fraction ≥ `rural_threshold`  
   - Urban fraction ≤ `urban_threshold`  
   - Located within `total_mask` (outside exclusion zone and within matrix bounds)  
   Cells failing any condition are set to `NaN`.  
4. **Output:**  
   The function returns three filtered submatrices: the primary data matrix, secondary matrix, and rural fraction matrix, all masked to include only valid rural candidate cells.  
### Spatial Design of Rural Reference Mask
For each urban grid cell, a rural reference mask is constructed within a rectangular neighborhood:
- **Inner rectangle (half-width `D1`)** – buffer zone around the urban cell to exclude nearby influenced cells  
- **Outer rectangle (half-width `D2`)** – extends the search for potential rural cells  
- **Annular region** – space between the inner and outer rectangles, from which rural reference cells are extracted  
The outer half-width `D2` is iteratively adjusted for each urban cell. Iteration stops when the ratio of filtered rural cells (`N_rural`) to total annular cells (`N_total`) meets a specified threshold (`ratio_lim`). This adapts the rural reference area to the scale of the urban zone, enhancing robustness across cities of different sizes.  
### Optional Altitude Filtering
To reduce biases from elevation differences, rural candidate cells can be excluded if their altitude differs from the urban cell by more than a user-defined threshold (`orograph_threshold`). This ensures temperature differences reflect urban influence rather than topography.  
### Total Candidate Cells in the Annular Region
The total number of grid cells in the annular region, before any filtering, is calculated as the difference between the areas of the outer and inner rectangles:
\[
N_{\text{total}} = (2 \times D_2 + 1)^2 - (2 \times D_1 + 1)^2
\]
Where:
- `D1` – half-width of the inner rectangle (exclusion zone around the urban cell)  
- `D2` – half-width of the outer rectangle (defines the search area for potential rural cells)  
The formula computes the total number of candidate cells in the annular region by subtracting the number of cells in the inner rectangle from the total number of cells in the outer rectangle. 


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
