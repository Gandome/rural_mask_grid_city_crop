
from climate_city_stats.utils import extract_population_on_climate_data_grid


raster_path = "/home/quenumm/Documents/data/urban_settlement_data/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif"
city_gpkg = "/home/quenumm/Documents/data/urban_settlement_data/GHS_UCDB_REGION_EUROPE_R2024A_V1_0/GHS_UCDB_REGION_EUROPE_R2024A.gpkg"
clim_nc_path = "/home/quenumm/Documents/belenos/data/Tmax/tasmax_ALPX-3_CNRM-ESM2-1_historical_r1i1p1f2_CNRM-MF_CNRM-AROME46t1_v1-r1_1hr_JJAS_1995.nc"
selected_cities = [
    "Paris", "Dijon", "Bordeaux", "Lille", "Clermont-Ferrand",
    "Montpellier", "Grenoble", "Strasbourg", "Lyon"
] ## this is an example

extract_population_on_climate_data_grid(
    raster_path=raster_path,
    city_gpkg=city_gpkg,
    clim_nc_path=clim_nc_path,
    output_dir="output_nc_files",
    country_code="France",
#    selected_cities=selected_cities,
    save=True
)

