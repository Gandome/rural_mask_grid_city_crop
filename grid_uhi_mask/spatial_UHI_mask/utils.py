import shutil
import numpy as np
import xarray as xr
import glob as gb
import os
import concurrent.futures
import time
import datetime
from multiprocessing import Pool
# import dask.array as da
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#######################################################
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

def extract_filtered_submatrix(matrix, matrix2, rural_frac, urban_frac, urban_point, nbg, nO, rural_threshold, urban_threshold):
    x, y = urban_point
    size = matrix.shape

    x_min, x_max = max(0, x - nbg), min(size[0], x + nbg + 1)
    y_min, y_max = max(0, y - nbg), min(size[1], y + nbg + 1)

    matrix_np = matrix if isinstance(matrix, np.ndarray) else matrix.values
    matrix2_np = matrix2 if isinstance(matrix2, np.ndarray) else matrix2.values
    rural_frac_np = rural_frac if isinstance(rural_frac, np.ndarray) else rural_frac.values
    urban_frac_np = urban_frac if isinstance(urban_frac, np.ndarray) else urban_frac.values

    submatrix = matrix_np[x_min:x_max, y_min:y_max].copy()
    submatrix2 = matrix2_np[x_min:x_max, y_min:y_max].copy()
    rural_sub = rural_frac_np[x_min:x_max, y_min:y_max].copy()
    urban_sub = urban_frac_np[x_min:x_max, y_min:y_max].copy()

    mask = np.ones(submatrix.shape, dtype=bool)
    inner_x_min, inner_x_max = nbg - nO, nbg + nO + 1

    inner_y_min, inner_y_max = nbg - nO, nbg + nO + 1

    if (0 <= inner_x_min < submatrix.shape[0]) and (0 <= inner_x_max <= submatrix.shape[0]) and \
       (0 <= inner_y_min < submatrix.shape[1]) and (0 <= inner_y_max <= submatrix.shape[1]):
        mask[inner_x_min:inner_x_max, inner_y_min:inner_y_max] = False

    valid_mask = np.zeros(submatrix.shape, dtype=bool)
    for i in range(submatrix.shape[0]):
        for j in range(submatrix.shape[1]):
            global_x = x_min + i
            global_y = y_min + j
            if (i != nbg or j != nbg) and (nO < global_x < size[0] - nO) and (nO < global_y < size[1] - nO):
                valid_mask[i, j] = True

    total_mask = mask & valid_mask
    condition_mask = (rural_sub >= rural_threshold) & (urban_sub <= urban_threshold) & total_mask

    filtered_matrix = submatrix.copy()
    filtered_matrix2 = submatrix2.copy()
    filtered_rural_frac = rural_sub.copy()

    filtered_matrix[~condition_mask] = np.nan
    filtered_matrix2[~condition_mask] = np.nan
    filtered_rural_frac[~condition_mask] = np.nan
    submatrix[~total_mask] = np.nan

    return submatrix, filtered_matrix, filtered_matrix2, filtered_rural_frac


# Function to calculate Urban Heat Island (UHI) for a given grid point
def uhi_formula(urban_temp, rural_temp):
    return urban_temp - rural_temp

def safe_nanmean(arr):
    return np.nanmean(arr) if not np.isnan(arr).all() else 0


def calculate_uhi(temperature, Elevation, rural_threshold, urban_threshold, rural_frac, urban_frac,
                  urban_grid_points, Min_Value, nbg, max_iterations, nO, height_lim1, height_lim2, height_lim3, height_lim4):

    def calculate_uhi_for_point(urban_point, nbg, rural_frac, urban_frac, temperature, Elevation, Min_Value,
                                max_iterations, nO, rural_threshold, urban_threshold, height_lim1, height_lim2, height_lim3, height_lim4):
        urban_point_tuple = tuple(urban_point)
        original_min_value = Min_Value

        while Min_Value >= 45:
            iterations = 0
            current_nbg = nbg  # Reset nbg for each Min_Value attempt
            
            while iterations < max_iterations:
                submatrix, temperatures_CC, Elevation_CC, rural_fract_CC = extract_filtered_submatrix(
                    temperature, Elevation, rural_frac, urban_frac,
                    urban_point_tuple, current_nbg, nO,
                    rural_threshold, urban_threshold
                )

                non_nan_count_submatrix = np.count_nonzero(~np.isnan(submatrix))
                non_nan_count_temperatures_CC = np.count_nonzero(~np.isnan(temperatures_CC))
                Ratio = 100 * non_nan_count_temperatures_CC / non_nan_count_submatrix if non_nan_count_submatrix > 0 else 0

                if Min_Value <= Ratio:
#                     temperatures_Elev = np.array(temperatures_CC) * np.array(Elevation_CC)

                    delta_elev = Elevation[urban_point_tuple] - Elevation_CC
                    delta_elev1 = delta_elev.copy()
                    delta_elev2 = delta_elev.copy()
                    delta_elev3 = delta_elev.copy()
                    delta_elev4 = delta_elev.copy()
                    delta_elev1[np.abs(delta_elev1) > height_lim1] = np.nan
                    delta_elev2[np.abs(delta_elev2) > height_lim2] = np.nan
                    delta_elev3[np.abs(delta_elev3) > height_lim3] = np.nan
                    delta_elev4[np.abs(delta_elev4) > height_lim4] = np.nan

                    rural_temp = np.nanmean(temperatures_CC)
                    rural_temp_Elev1 = rural_temp -0.0065*safe_nanmean(delta_elev1)
                    rural_temp_Elev2 = rural_temp -0.0065*safe_nanmean(delta_elev2)
                    rural_temp_Elev3 = rural_temp -0.0065*safe_nanmean(delta_elev3)
                    rural_temp_Elev4 = rural_temp -0.0065*safe_nanmean(delta_elev4)
                    #np.nanmean(np.array(temperatures_CC) * np.array(Elevation_CC))
#                     rural_temp_Elev = np.nanmean(np.array(temperatures_CC) * np.array(Elevation_CC))

                    urban_temp = temperature[urban_point[0], urban_point[1]]

                    uhi1 = uhi_formula(urban_temp, rural_temp)
                    uhi_elv1 = uhi_formula(urban_temp, rural_temp_Elev1)
                    uhi_elv2 = uhi_formula(urban_temp, rural_temp_Elev2)
                    uhi_elv3 = uhi_formula(urban_temp, rural_temp_Elev3)
                    uhi_elv4 = uhi_formula(urban_temp, rural_temp_Elev4)


                    return uhi1, uhi_elv1, uhi_elv2, uhi_elv3, uhi_elv4, rural_temp, rural_temp_Elev1, rural_temp_Elev2, rural_temp_Elev3, rural_temp_Elev4, Ratio, current_nbg
                else:
                    current_nbg += 1  # Expand the neighborhood range

                iterations += 1

            # If we’re here, the Ratio was still too low, so reduce Min_Value
            Min_Value -= 5

        # If no valid rural points are found after all Min_Value attempts
        return None, None, None,None, None, None, None, None, None, None, Ratio, current_nbg
#     print(Min_Value)

    # We initialize arrays to store UHI values and rural grid properties
    uhi_values1 = np.zeros_like(temperature)
    uhi_Elev1 = np.zeros_like(temperature)
    uhi_Elev2 = np.zeros_like(temperature)
    uhi_Elev3 = np.zeros_like(temperature)
    uhi_Elev4 = np.zeros_like(temperature)
    rural_mean_values = np.zeros_like(temperature)
    rural_mean_Elev1 = np.zeros_like(temperature)
    rural_mean_Elev2 = np.zeros_like(temperature)
    rural_mean_Elev3 = np.zeros_like(temperature)
    rural_mean_Elev4 = np.zeros_like(temperature)
    Min_Value_values = np.zeros_like(temperature)
    current_nbg_values = np.zeros_like(temperature)

    #nb_urb_grd = 0
    for urban_point in urban_grid_points:
        result = calculate_uhi_for_point(urban_point, nbg, rural_frac, urban_frac, temperature, Elevation,
                                         Min_Value, max_iterations, nO, rural_threshold, urban_threshold, height_lim1, height_lim2, height_lim3, height_lim4)

        if result:  # Only assign results if they are not None
            uhi_values1[urban_point[0], urban_point[1]], uhi_Elev1[urban_point[0], urban_point[1]], \
            uhi_Elev2[urban_point[0], urban_point[1]], uhi_Elev3[urban_point[0], urban_point[1]], \
            uhi_Elev4[urban_point[0], urban_point[1]], rural_mean_values[urban_point[0], urban_point[1]], \
            rural_mean_Elev1[urban_point[0], urban_point[1]], rural_mean_Elev2[urban_point[0], urban_point[1]], \
            rural_mean_Elev3[urban_point[0], urban_point[1]], rural_mean_Elev4[urban_point[0], urban_point[1]],\
            Min_Value_values[urban_point[0], urban_point[1]], current_nbg_values[urban_point[0], urban_point[1]] = result
#         print(Min_Value_values)
    # Return the calculated values
    return uhi_values1, uhi_Elev1, uhi_Elev2, uhi_Elev3, uhi_Elev4, rural_mean_values, rural_mean_Elev1, \
            rural_mean_Elev2, rural_mean_Elev3, rural_mean_Elev4, Min_Value_values, current_nbg_values

