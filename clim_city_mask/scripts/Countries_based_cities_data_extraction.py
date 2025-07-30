
from climate_city_stats.utils import kelvin_humidity_convert, sanitize_filename, format_units, build_grid_polygons, group_time_data, compute_city_stats, run_city_climate_analysis
from climate_city_stats.io import load_city_polygons, load_climate_data

######*************
file_gpkg = "/home/quenumm/Documents/data/urban_settlement_data/GHS_UCDB_REGION_EUROPE_R2024A_V1_0/GHS_UCDB_REGION_EUROPE_R2024A.gpkg"
data_file = "/home/quenumm/Documents/belenos/data/Tmax/tasmax_ALPX-3_CNRM-ESM2-1_historical_r1i1p1f2_CNRM-MF_CNRM-AROME46t1_v1-r1_1hr_JJAS_1995.nc"

run_city_climate_analysis(
    file_gpkg=file_gpkg,
    data_file=data_file,
    variable="tas", # must be updated
    target_countries=["France"],        
    target_city_names=["Paris", "Lyon"],
    time_periods=["seasonal"],
    plot=True,
    save_masked_netcdf=True,
    cmap="plasma"
)


