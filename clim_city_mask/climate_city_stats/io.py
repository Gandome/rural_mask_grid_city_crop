import xarray as xr
import geopandas as gpd


def load_city_polygons(gpkg_path, crs="EPSG:4326", target_names=None):
    gdf = gpd.read_file(gpkg_path).to_crs(crs)
    if target_names:
        gdf = gdf[gdf['GC_UCN_MAI_2025'].isin(target_names)]
    return gdf

def load_climate_data(nc_file, variable):
    ds = xr.open_dataset(nc_file)
    return ds, ds[variable]

