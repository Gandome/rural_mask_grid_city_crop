import os
import re
import numpy as np
import geopandas as gpd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point
from tqdm import tqdm
import glob as gb
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
from rasterio.mask import mask
import rioxarray
from shapely import contains

from climate_city_stats.io import load_city_polygons, load_climate_data 

#################***********************************************************
#def sanitize_filename(name):
#    return name.replace(" ", "_").replace("/", "-")
def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)

def kelvin_humidity_convert(ds, variable):
    units = ds[variable].attrs.get('units', '').lower()
    if units == 'k':
        ds[variable] = ds[variable] - 273.15
        ds[variable].attrs['units'] = 'degC'
    elif units == 'g/kg':
        ds[variable] = ds[variable] / 1000.0
        ds[variable].attrs['units'] = 'kg/kg'
    elif units == 'kg/kg':
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs['units'] = 'g/kg'
    return ds

def format_units(units):
    units = units.lower()
    return {
        'degc': '°C',
        'k': 'K',
        'g/kg': 'gkg⁻¹',
        'kg/kg': 'kgkg⁻¹'
    }.get(units, units)
    
def build_grid_polygons(lon2d, lat2d):
    ny, nx = lon2d.shape
    cell_polys = np.empty((ny, nx), dtype=object)
    for j in range(ny - 1):
        for i in range(nx - 1):
            lon_corners = [lon2d[j, i], lon2d[j, i+1], lon2d[j+1, i+1], lon2d[j+1, i]]
            lat_corners = [lat2d[j, i], lat2d[j, i+1], lat2d[j+1, i+1], lat2d[j+1, i]]
            cell_polys[j, i] = Polygon(zip(lon_corners, lat_corners))
    return cell_polys

def group_time_data(var_data, period):
    if period == "daily":
        return var_data.resample(time="1D")
    elif period == "monthly":
        return var_data.resample(time="1MS")
    elif period == "seasonal":
        return var_data.resample(time="QS-DEC")
    else:
        raise ValueError("Invalid time period: choose from 'daily', 'monthly', 'seasonal'")
        
# ========== Selected city Data Cropping ==========
def crop_city_data(city, var_data, lon2d, lat2d, variable_list, frac_threshold=0.01):
    ny, nx = lon2d.shape
    grid_polys = build_grid_polygons(lon2d, lat2d)

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
                if frac > frac_threshold:
                    frac_mask[j, i] = frac

    if frac_mask.sum() == 0:
        print(f"Warning: {city_name} has no overlapping grid cells. Skipping.")
        return None

    rows, cols = np.where(frac_mask > 0)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    frac_crop = frac_mask[row_min:row_max+1, col_min:col_max+1]
    lon_crop = lon2d[row_min:row_max+1, col_min:col_max+1]
    lat_crop = lat2d[row_min:row_max+1, col_min:col_max+1]

    var_dict = {}
    for v in variable_list:
        arr = var_data[v][:, row_min:row_max+1, col_min:col_max+1]
        mask_3d = np.broadcast_to((frac_crop == 0)[None, :, :], arr.shape)
        masked_array = np.ma.masked_array(arr.values, mask=mask_3d)
        filled_array = np.where(masked_array.mask, np.nan, masked_array.data)
        var_dict[v] = (['time', 'y', 'x'], filled_array)

    ds_city = xr.Dataset(
        var_dict,
        coords={
            'time': var_data[variable_list[0]].time,
            'lat': (['y', 'x'], lat_crop),
            'lon': (['y', 'x'], lon_crop)
        },
        attrs={
            'city_name': city_name,
            'country': country,
            'history': f"Cropped hourly time series to {city_name}"
        }
    )

    for v in variable_list:
        ds_city[v].attrs['units'] = var_data[v].attrs.get("units", "")

    return ds_city


# # #=======> ***** <======
def plot_city_and_rural_overlay(city_overlay, rural_mask, var_data, city_name, R, title="City and Rural Map", save_path=None, plot_city_overlay=True):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get rural_mask boolean array
    mask = rural_mask["rural_mask"].values == 1
    
#     mask = var_data["UHIpx"].isel(time=0).where(rural_mask["rural_mask"] == 1)

    # bounding box (y_min:y_max+1, x_min:x_max+1)
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("No rural_mask == 1 found. Cannot plot.")

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Separate rural_mask and city_overlay
    cropped_rural = rural_mask["rural_mask"].isel(y=slice(y_min, y_max+1), x=slice(x_min, x_max+1))
    cropped_lat = rural_mask["lat"].isel(y=slice(y_min, y_max+1), x=slice(x_min, x_max+1))
    cropped_lon = rural_mask["lon"].isel(y=slice(y_min, y_max+1), x=slice(x_min, x_max+1))
    cropped_city = city_overlay[y_min:y_max+1, x_min:x_max+1]

    # cropping the var_data["UHIpx"] to the same box as the mask
    cropped_var = var_data["UHIpx"].isel(time=0, y=slice(y_min, y_max+1), x=slice(x_min, x_max+1))
    masked = cropped_var.where(cropped_rural == 1)

    # extent
    extent = [cropped_lon.min(), cropped_lon.max(), cropped_lat.min(), cropped_lat.max()]
    origin = 'lower' if cropped_lat[0, 0] < cropped_lat[-1, 0] else 'upper'

    # condition for shared color range
    if plot_city_overlay:
        vmin = min(np.nanmin(cropped_city), np.nanmin(masked))
        vmax = max(np.nanmax(cropped_city), np.nanmax(masked))*1.05
    else:
        
        vmin, vmax = np.nanmin(masked), np.nanmax(masked)

    # Plot
    rm = ax.imshow(masked, origin=origin, extent=extent, cmap='Greens', alpha=0.8, vmin=vmin, vmax=vmax)
    im = None
    if plot_city_overlay:
        im = ax.imshow(cropped_city, origin=origin, extent=extent, cmap='coolwarm', alpha=1.0, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # side-by-side colorbars
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)

    if plot_city_overlay and im is not None:
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(im, cax=cax1)
#         cbar1.set_label("UHI (°C)")
        cbar1.ax.set_yticklabels([])  # Hide tick labels for city overlay
        cbar2 = fig.colorbar(rm, cax=cax2)
        cbar2.set_label("UHI (°C)")
    else:
        cbar = fig.colorbar(rm, cax=cax1)
        cbar.set_label("UHI (°C)")
        
    # ===labels 'city' and 'rural area' ===
    def compute_centroid(mask_arr, lon_arr, lat_arr):
        yx = np.argwhere(mask_arr)
        if len(yx) == 0:
            return None, None
        y_coords, x_coords = yx[:, 0], yx[:, 1]
        lat_centroid = np.mean(lat_arr[y_coords, x_coords])
        lon_centroid = np.mean(lon_arr[y_coords, x_coords])
        return lon_centroid, lat_centroid

    # City label: center
    lon_city, lat_city = compute_centroid(~np.isnan(cropped_city), cropped_lon.values, cropped_lat.values)
    if lon_city is not None:
        ax.text(lon_city, lat_city, "city", color="red", fontsize=12, weight="bold",
                ha="center", va="center", bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

    # Rural area label: top-right
    lon_rural_corner = cropped_lon.values[int(R/2), -int(R/2)]  # 
    lat_rural_corner = cropped_lat.values[int(R/2), -int(R/2)]
    ax.text(lon_rural_corner, lat_rural_corner, "rural area", color="green", fontsize=12, weight="bold",
            ha="right", va="top", bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.3'))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

###
def get_city_offset_indices(city_ds, var_data):
    city_latlon = np.column_stack((city_ds.lat.values.ravel(), city_ds.lon.values.ravel()))
    var_latlon = np.column_stack((var_data.lat.values.ravel(), var_data.lon.values.ravel()))
    tree = cKDTree(var_latlon)
    dist, flat_indices = tree.query(city_latlon)
    indx_y, indx_x = np.unravel_index(flat_indices, var_data.lat.shape)
    return indx_y.reshape(city_ds.lat.shape), indx_x.reshape(city_ds.lon.shape)

def create_city_overlay_from_indices(city_ds, var_data, indx_y, indx_x, varname='UHIpx'):
    ny_global, nx_global = var_data.sizes['y'], var_data.sizes['x']
    city_data = city_ds[varname].isel(time=0).values
    overlay = np.full((ny_global, nx_global), np.nan)
    for y in range(city_data.shape[0]):
        for x in range(city_data.shape[1]):
            val = city_data[y, x]
            if np.isnan(val):
                continue
            gy, gx = indx_y[y, x], indx_x[y, x]
            if 0 <= gy < ny_global and 0 <= gx < nx_global:
                overlay[gy, gx] = val
    return overlay
##

def extract_rural_mask_global(var_data, city_ds, indx_y, indx_x):
    UHIpx_2d = city_ds['UHIpx'].isel(time=0).values
    nbg_2d = city_ds['nbg'].isel(time=0).values
    ny_global, nx_global = var_data.sizes['y'], var_data.sizes['x']
    lon2d = var_data['lon'].values
    lat2d = var_data['lat'].values
    rural_mask = np.zeros((ny_global, nx_global), dtype=int)

    for y in range(UHIpx_2d.shape[0]):
        for x in range(UHIpx_2d.shape[1]):
            if np.isnan(UHIpx_2d[y, x]):
                continue
            r = nbg_2d[y, x]
            if np.isnan(r) or r == 0:
                continue
            cy, cx = int(indx_y[y, x]), int(indx_x[y, x])
            r = int(r)
            top = max(0, cy - r)
            bottom = min(ny_global - 1, cy + r)
            left = max(0, cx - r)
            right = min(nx_global - 1, cx + r)
            for yy in range(top, bottom + 1):
                for xx in range(left, right + 1):
                    if yy == cy and xx == cx:
                        continue
                    rural_mask[yy, xx] = 1

    # Here we mask out all valid city pixels from the global rural mask
    for y in range(UHIpx_2d.shape[0]):
        for x in range(UHIpx_2d.shape[1]):
            if np.isnan(UHIpx_2d[y, x]):
                continue
            cy, cx = int(indx_y[y, x]), int(indx_x[y, x])
            rural_mask[cy, cx] = 0

    return r, xr.Dataset(
        data_vars={'rural_mask': (('y', 'x'), rural_mask)},
        coords={'lat': (('y', 'x'), lat2d), 'lon': (('y', 'x'), lon2d)},
        attrs={'description': 'Rural pixel mask excluding city grid cells'}
    )

def extract_population_on_climate_data_grid(
    raster_path: str,
    city_gpkg: str,
    clim_nc_path: str,
    output_dir: str = "population_climate_outputs",
    country_code: str = None,
    selected_cities: list = None,
    save: bool = True
):
    """
    For each city in selected_cities, extract fine-grid population and aggregate into the climate grid.

    Parameters
    ----------
    raster_path : str
        Path to population raster (.tif).
    city_gpkg : str
        Path to city boundaries geopackage (.gpkg).
    clim_nc_path : str
        Path to climate NetCDF file with coarse lat/lon grid.
    output_dir : str
        Directory for saving per-city NetCDFs.
    country_code : str, optional
        Country code to filter cities.
    selected_cities : list of str, this also is optional
        List of city names to extract.
    save : bool
        If True, saves the NetCDF output per city.
    """
    os.makedirs(output_dir, exist_ok=True)

    # base climate dataset 
    base_clim_ds = xr.open_dataset(clim_nc_path)

    # city boundaries
    city_gdf = gpd.read_file(city_gpkg)
    if country_code:
        city_gdf = city_gdf[city_gdf["GC_CNT_GAD_2025"] == country_code]
    if selected_cities:
        city_gdf = city_gdf[city_gdf["GC_UCN_MAI_2025"].isin(selected_cities)]

    print(f"Found {len(city_gdf)} cities to process.")

    with rasterio.open(raster_path) as src:
        city_gdf = city_gdf.to_crs(src.crs)

        for _, row in city_gdf.iterrows():
            city_name = row["GC_UCN_MAI_2025"].replace("/", "_").replace(" ", "_")
            print(f"Processing city: {city_name}")
            geom = [row.geometry]

            try:
                # mask of population raster
                out_image, out_transform = mask(src, geom, crop=True)
                data = out_image[0]
                data = np.where(data == src.nodata, np.nan, data)

                ny, nx = data.shape
                x = np.arange(nx) * out_transform.a + out_transform.c + out_transform.a / 2
                y = np.arange(ny) * out_transform.e + out_transform.f + out_transform.e / 2

                pop_da = xr.DataArray(data, coords={"y": y, "x": x}, dims=("y", "x"))
                pop_da = pop_da.rio.write_crs("ESRI:54009")
                pop_wgs84 = pop_da.rio.reproject("EPSG:4326")

                # 
                lon_fine, lat_fine = np.meshgrid(pop_wgs84["x"].values, pop_wgs84["y"].values)
                pop_flat = pop_wgs84.values.flatten()
                lon_flat = lon_fine.flatten()
                lat_flat = lat_fine.flatten()

                # 
                clim_ds = base_clim_ds.copy()
                lat_coarse = clim_ds["lat"].values  # shape (y, x)
                lon_coarse = clim_ds["lon"].values

                if lat_coarse.ndim != 2 or lon_coarse.ndim != 2:
                    raise ValueError("Climate dataset must use 2D lat/lon grids.")

                n_y, n_x = lat_coarse.shape
                pop_sum_coarse = np.zeros((n_y, n_x))

                # Assignment of population into climate grid
                lat_flat = lat_flat.astype(np.float64)
                lon_flat = lon_flat.astype(np.float64)
                valid = ~np.isnan(pop_flat)

                for lat_val, lon_val, val in zip(lat_flat[valid], lon_flat[valid], pop_flat[valid]):
                    # Identificatin of the nearest climate grid cell
                    dist2 = (lat_coarse - lat_val) ** 2 + (lon_coarse - lon_val) ** 2
                    i, j = np.unravel_index(np.argmin(dist2), dist2.shape)
                    pop_sum_coarse[i, j] += val

                # Cropping of to the region where population exist
                mask_pop = pop_sum_coarse > 0
                if not np.any(mask_pop):
                    raise ValueError("No population found in climate grid overlap.")

                ys, xs = np.where(mask_pop)
                min_y, max_y = ys.min(), ys.max()
                min_x, max_x = xs.min(), xs.max()

                pop_sum_cropped = pop_sum_coarse[min_y:max_y+1, min_x:max_x+1]
                clim_ds_cropped = clim_ds.isel(y=slice(min_y, max_y+1), x=slice(min_x, max_x+1))

                # Adding to the cropped data the population variable
                clim_ds_cropped["population"] = (("y", "x"), pop_sum_cropped)
                clim_ds_cropped["population"].attrs = {
                    "units": "people per grid cell",
                    "long_name": f"Population aggregated for {city_name}"
                }

                if save:
                    out_nc = os.path.join(output_dir, f"Population_{city_name}.nc")
                    clim_ds_cropped.to_netcdf(out_nc)
                    print(f"Saved: {out_nc}")

            except Exception as e:
                print(f"Error processing {city_name}: {e}")

def compute_city_stats(
    cities_gdf, var_data, lon2d, lat2d, variable,
    plot_dir="plots", plot=False,
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

            lat_center = lat_crop
            lon_center = lon_crop

            point_mask = np.zeros_like(frac_crop, dtype=bool)
            for j in range(frac_crop.shape[0]):
                for i in range(frac_crop.shape[1]):
                    point = Point(lon_center[j, i], lat_center[j, i])
                    point_mask[j, i] = polygon.contains(point)

            combined_mask = (frac_crop == 0) | (~point_mask)
            masked_vals = np.ma.masked_array(vals, mask=combined_mask)

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
                plt.close()

        if save_masked_netcdf and city_data_list:
            full_array = np.stack(city_data_list, axis=0)
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

# ==================

def run_city_climate_analysis(
    file_gpkg,
    data_file,
    variable,
    target_countries=None,
    target_city_names=None,
    time_periods=['seasonal'],
    plot=True,
    save_masked_netcdf=True,
    cmap='viridis'
):
    if isinstance(time_periods, str):
        time_periods = [time_periods]

    targets = target_city_names if target_city_names else target_countries
    if not targets:
        raise ValueError("You must specify either target_countries or target_city_names.")

    for target in targets:
        for time_period in time_periods:
            label = sanitize_filename(target)
            plot_dir = f"city_plots_{label}_{time_period}"
            output_csv = f"city_temperature_statistics_{label}_{time_period}.csv"
            netcdf_dir = f"masked_vals_{label}_{time_period}"

            cities_gdf = load_city_polygons(
                file_gpkg,
                target_names=[target] if target_city_names else None,
                target_country=target if target_countries else None
            )

            data, var_data = load_climate_data(data_file, variable)
            data = kelvin_humidity_convert(data, variable)
            var_data = data[variable]
            lon2d = data["lon"].values
            lat2d = data["lat"].values

            compute_city_stats(
                cities_gdf, var_data, lon2d, lat2d, variable,
                plot_dir=plot_dir, plot=plot,
                time_period=time_period,
                output_csv=output_csv,
                save_masked_netcdf=save_masked_netcdf,
                netcdf_dir=netcdf_dir, cmap=cmap
            )




