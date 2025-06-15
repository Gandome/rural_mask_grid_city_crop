import os
import numpy as np
import geopandas as gpd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm


def sanitize_filename(name):
    return name.replace(" ", "_").replace("/", "-")

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

