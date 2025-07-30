import glob as gb
import os
import xarray as xr
from tqdm import tqdm

###########################

from climate_city_stats.utils import sanitize_filename, build_grid_polygons, crop_city_data, plot_city_and_rural_overlay, get_city_offset_indices, create_city_overlay_from_indices, extract_rural_mask_global
from climate_city_stats.io import load_city_polygons


# ========== Execution Block for extraction ==========
if __name__ == "__main__":
    path_uhi1 = "/home/quenumm/Documents/belenos/data/NFR010d"
    file_gpkg = "/home/quenumm/Documents/data/urban_settlement_data/GHS_UCDB_REGION_EUROPE_R2024A_V1_0/GHS_UCDB_REGION_EUROPE_R2024A.gpkg"
    target_cities = ["Paris", "Dijon", "Rouen", "Caen", "Amiens", "Orleans", "Troyes", "Reims"]
    variable_list = ['UHIpx', 'UHIpx_mean', 'nbg']
    simulation = ['2020_2022']
    fle = 'Urban_Heat_Island_data_NFR2.5_ERA5_evaluation_r1i1p1f1_CNRM-MF_CNRM-AROME46t1_v1-r1_1hr_2022_000340.nc'

    cities_gdf = load_city_polygons(file_gpkg, target_names=target_cities)
    data_path = gb.glob(os.path.join(path_uhi1, '2020_2022', fle))
    if not data_path:
        raise FileNotFoundError("NetCDF file not found.")
    data = xr.open_dataset(data_path[0])
    var_data = data[variable_list]
    lon2d = data["lon"].values
    lat2d = data["lat"].values

    for simu in simulation:
        for _, city in tqdm(cities_gdf.iterrows(), total=len(cities_gdf), desc=f"Processing {simu}"):
            city_name = sanitize_filename(city['GC_UCN_MAI_2025'])
            print(f'City: {city_name}')

            city_ds = crop_city_data(city, var_data, lon2d, lat2d, variable_list)
            if city_ds is None:
                continue

            indx_y, indx_x = get_city_offset_indices(city_ds, var_data)
            R, rural_mask = extract_rural_mask_global(var_data, city_ds, indx_y, indx_x)
            city_overlay = create_city_overlay_from_indices(city_ds, var_data, indx_y, indx_x, varname='UHIpx')

            out_path = f"output/UHI_city_{city_name}_{simu}.nc"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            city_ds.to_netcdf(out_path)
            out_path2 = f"output/rural_of_city_{city_name}_{simu}.nc"
            os.makedirs(os.path.dirname(out_path2), exist_ok=True)
            rural_mask.to_netcdf(out_path2)
            
            plot_path = f"figures/UHI_overlay_{city_name}_{simu}.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plot_city_and_rural_overlay(
                city_overlay, rural_mask, var_data, city_name, R, 
                title=f"UHI over {city_name}" , #;  with nbg:  {int(city_ds.nbg.max().values.max())}",
                save_path=plot_path, plot_city_overlay=True)

            
