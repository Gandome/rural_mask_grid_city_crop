from climate_city_stats.io import load_city_polygons, load_climate_data
from climate_city_stats.utils import kelvin_humidity_convert, sanitize_filename, format_units, build_grid_polygons, group_time_data
from climate_city_stats.stats import compute_city_stats

file_gpkg = "/home/quenumm/Documents/Scripts/Python/github/UHI_grid_mask-city/rural_mask_grid_city_crop/clim_city_mask/data/GHS_UCDB_REGION_EUROPE_R2024A.gpkg"
data_file = "/home/quenumm/Documents/data/data_lea/2022/day/output/Urban_Heat_Island_data_NFR2.5_ERA5_eval_CNRM-AROME46t1_v1_1hr_2022_000001.nc"
target_cities = ["Paris"]# , "Berlin", "Rome"]
variable = "UHI_LR100"
time_period=["seasonal"] #, 'seasonal', 'daily', 'monthly'
cmap='viridis'

for time_period in time_period:
    plot_dir=f"plots/{time_period}"
    output_csv=f"results/city_statistics_{time_period}.csv"
    netcdf_dir=f"outputs/masked_vals_{time_period}"

    cities_gdf = load_city_polygons(file_gpkg, target_names=target_cities)
    data, var_data = load_climate_data(data_file, variable)
    data = kelvin_humidity_convert(data, variable)
    var_data = data[variable]
    lon2d = data["lon"].values
    lat2d = data["lat"].values

    compute_city_stats(
        cities_gdf, var_data, lon2d, lat2d, variable,
        plot_dir=plot_dir, plot=True,
        time_period=time_period,
        output_csv=output_csv,
        save_masked_netcdf=True,
        netcdf_dir=netcdf_dir,
        cmap=cmap
    )

