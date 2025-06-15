import os
import numpy as np
import geopandas as gpd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm
from climate_city_stats.utils import build_grid_polygons, group_time_data, format_units, sanitize_filename


def compute_city_stats(
    cities_gdf, var_data, lon2d, lat2d, variable, plot_dir="plots", plot=False,
    time_period="seasonal", output_csv="city_statistics.csv",
    save_masked_netcdf=False, netcdf_dir="masked_vals_netcdf", cmap=False):

    ny, nx = lon2d.shape
    grid_polys = build_grid_polygons(lon2d, lat2d)
    os.makedirs(plot_dir, exist_ok=True)
    if save_masked_netcdf:
        os.makedirs(netcdf_dir, exist_ok=True)
    results = []

    for idx, city in tqdm(cities_gdf.iterrows(), total=len(cities_gdf)):
        city_name = city['GC_UCN_MAI_2025']
        country = city['GC_CNT_GAD_2025']
        polygon = city['geometry'].buffer(0)

        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda a: a.area)

        frac_mask = np.zeros((ny, nx))
        for j in range(ny - 1):
            for i in range(nx - 1):
                cell_poly = grid_polys[j, i]
                if cell_poly and polygon.intersects(cell_poly):
                    inter = polygon.intersection(cell_poly)
                    frac = inter.area / cell_poly.area if cell_poly.area > 0 else 0
                    frac_mask[j, i] = frac

        if frac_mask.sum() == 0:
            print(f"Warning: {city_name} has no overlapping grid cells. Skipping.")
            continue

        rows, cols = np.where(frac_mask > 0)
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()

        frac_crop = frac_mask[row_min:row_max+1, col_min:col_max+1]
        lon_crop = lon2d[row_min:row_max+1, col_min:col_max+1]
        lat_crop = lat2d[row_min:row_max+1, col_min:col_max+1]

        time_groups = group_time_data(var_data, time_period)
        time_list = []
        city_data_list = []

        for t_name, t_group in time_groups:
            t_group_crop = t_group[:, row_min:row_max+1, col_min:col_max+1]
            vals = t_group_crop.mean(dim="time").values
            masked_vals = np.ma.masked_array(vals, mask=(frac_crop == 0))

            mean_val = np.ma.average(masked_vals, weights=frac_crop)
            median_val = np.ma.median(masked_vals)
            std_val = np.ma.std(masked_vals)
            min_val = masked_vals.min()
            max_val = masked_vals.max()
            n_cells = (frac_crop > 0).sum()
            total_weight = frac_crop.sum()
            units_raw = var_data.attrs.get("units", "")
            units_fmt = format_units(units_raw)

            results.append({
                "City": city_name,
                "Country": country,
                "Period": pd.to_datetime(t_name).strftime("%Y-%m-%d"),
                "Mean": round(mean_val, 2),
                "Median": round(median_val, 2),
                "Mean - Median": round(mean_val - median_val, 2),
                "Std": round(std_val, 2),
                "Min": round(min_val, 2),
                "Max": round(max_val, 2),
                "Unit": units_fmt,
                "GridCells": int(n_cells),
                "TotalWeight": round(total_weight, 2)
            })

            city_data_list.append(np.where(masked_vals.mask, np.nan, masked_vals.data))
            time_list.append(pd.to_datetime(t_name))

            if plot:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.pcolormesh(
                    lon_crop, lat_crop, masked_vals,
                    shading='auto', cmap=cmap
                )
                gpd.GeoSeries(polygon).boundary.plot(ax=ax, color='cyan', linewidth=2)
                plt.colorbar(im, ax=ax, label=f'{variable} ({units_fmt})')
                plt.title(f"City of {city_name} on {pd.to_datetime(t_name).strftime('%Y-%m-%d')}")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.tight_layout()
                filename = f"{sanitize_filename(city_name)}_{sanitize_filename(country)}_{pd.to_datetime(t_name).strftime('%Y%m%d')}.png"
                filepath = os.path.join(plot_dir, filename)
                plt.savefig(filepath)
                #plt.show()

        if save_masked_netcdf and city_data_list:
            full_array = np.stack(city_data_list, axis=0)  # (time, y, x)
            ds_city = xr.Dataset(
                {
                    variable: (['time', 'y', 'x'], full_array)
                },
                coords={
                    'time': pd.DatetimeIndex(time_list),
                    'lat': (['y', 'x'], lat_crop),
                    'lon': (['y', 'x'], lon_crop)
                }
            )
            ds_city[variable].attrs['units'] = units_fmt
            ds_city.attrs['city_name'] = city_name
            ds_city.attrs['country'] = country
            ds_city.attrs['history'] = f"Cropped time series to {city_name}"

            filename = f"{sanitize_filename(city_name)}_{sanitize_filename(country)}_{time_period}.nc"
            output_path = os.path.join(netcdf_dir, filename)
            ds_city.to_netcdf(output_path)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to: {output_csv}")
    return df

