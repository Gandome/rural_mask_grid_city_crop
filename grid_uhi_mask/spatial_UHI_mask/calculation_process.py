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

from spatial_UHI_mask.utils import kelvin_humidity_convert, format_units, calculate_uhi
# from .parallel_process import process_folder_parallel
# from spatial_UHI_mask.urban_mask import classify_grid_points


# Function to process a single file
def process_file(file_path, Elevation, output_path, sea_mask, rural_threshold, urban_threshold, rural_frac, urban_frac,
                 urban_grid_points, Min_Value, nbg, max_iterations, nO, height_lim1, height_lim2, height_lim3, height_lim4):
    try:
        filename = os.path.basename(file_path)
       # print(f'the current treat file is {filename}')
        temperature_data = xr.open_dataset(file_path, chunks={'time': 'auto'})
        temperature_data['tas'] -= 273.15
        temperature_data = temperature_data.squeeze()
        temperature = temperature_data['tas']

       # print(f'dims of data: {temperature.shape}')
        lon = temperature.lon.values #[0, :]
        lat = temperature.lat.values #[:, 0]
        lon_bnds = temperature_data['lon_bnds'].values
        lat_bnds = temperature_data['lat_bnds'].values
        height = [2.0]

#         calculate_uhi(temperature, rural_threshold, urban_threshold, rural_frac, urban_frac, urban_grid_points, Min_Value, nbg, max_iterations, nO)
        def calculate_uhi_for_time_step(t):
            return calculate_uhi(temperature[t].values, Elevation.values, rural_threshold, urban_threshold, rural_frac.values,
                                 urban_frac.values, urban_grid_points,  Min_Value, nbg, max_iterations, nO,
                                 height_lim1, height_lim2, height_lim3, height_lim4)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            uhi_results = list(executor.map(calculate_uhi_for_time_step, range(temperature.shape[0])))

        uhi_data1, uhi_data_elv1, uhi_data_elv2, uhi_data_elv3, uhi_data_elv4, rural_mean_data, \
        rural_mean_data_elv1, rural_mean_data_elv2, rural_mean_data_elv3, rural_mean_data_elv4, \
        Min_Value, current_nbg = zip(*uhi_results)

        uhi_data1 = np.where(sea_mask.values, np.nan, np.array(uhi_data1))
        uhi_data_elv1 = np.where(sea_mask.values, np.nan, np.array(uhi_data_elv1))
        uhi_data_elv2 = np.where(sea_mask.values, np.nan, np.array(uhi_data_elv2))
        uhi_data_elv3 = np.where(sea_mask.values, np.nan, np.array(uhi_data_elv3))
        uhi_data_elv4 = np.where(sea_mask.values, np.nan, np.array(uhi_data_elv4))
        ###************
        rural_mean_data = np.where(sea_mask.values, np.nan, np.array(rural_mean_data))
        rural_mean_data_elv1 = np.where(sea_mask.values, np.nan, np.array(rural_mean_data_elv1))
        rural_mean_data_elv2 = np.where(sea_mask.values, np.nan, np.array(rural_mean_data_elv2))
        rural_mean_data_elv3 = np.where(sea_mask.values, np.nan, np.array(rural_mean_data_elv3))
        rural_mean_data_elv4 = np.where(sea_mask.values, np.nan, np.array(rural_mean_data_elv4))
        ####        
        Min_Value_values = np.where(sea_mask.values, np.nan, np.array(Min_Value))
        current_nbg_values = np.where(sea_mask.values, np.nan, np.array(current_nbg))

        uhi_attrs = {'long_name': 'Urban Heat Island (UHI)',
                     'units': 'degree Celsius',
                     'description': 'Urban Heat Island effect data',
                     'creation_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     'corresponding_name': 'Mayeul Quenum',
                     'contact': '+33(0)561079054',
                     'email': 'mayeul.quenum@meteo.fr',
                     'frequency': '1hr',
                     'institute_id': 'CNRM',
                     'model_id': 'CNRM-AROME46t1',
                     'project_id': 'IDF-CONF',
                     'domain': 'NFR2.5',
                     'rcm_version_id': 'v1',
                     'nesting_level': '1',
                     'comment_nesting': ('These are results of the 1st nest of a 1-nest-approach;\n'
                                         'CNRM-AROME CP-RCM is driven by ECMWF-ERA5 reanalysis'),
                     'comment_1nest': ('IDF configuration NFR2.5 CNRM-AROME46t1 ECMWF-ERA5 v1 : NFR009b.\n'
                                       'Ref: Caillaud et al. (2021). https://doi.org/10.1007/s00382-020-05558-y'),
                     'nominal_resolution': '2.5km',
                     'grid': 'regional-zone : C',
                     'grid_mapping_name': 'lambert conformal conic',
                     'longitude_of_central_meridian': '2.34f',
                     'standard_parallel': '48.85f',
                     'latitude_of_projection_origin': '48.85f',

                     'product': 'output',
                     'references': 'https://www.cnrm.meteo.fr/spip.php?article1094&lang=en',
                     'filename': (f"Urban Heat Island data for {filename[4:]}:\n"
                                  f"- Rural grid threshold: >= {rural_threshold}\n"
                                  f"- Urban grid threshold: >= {urban_threshold}\n"
                                  f"- Minimum distance of the rural grids to the city: {2.5 * nO} km"),
                     'CDO': 'Climate Data Operators version 2.0.4 (https://mpimet.mpg.de/cdo)'

                }

        # Define compression encoding for variables
        comp = dict(zlib=True, complevel=9)
        encoding = {
             var: comp for var in [
                 'UHI_px', f'UHI_LR{height_lim1}', f'UHI_LR{height_lim2}', f'UHI_LR{height_lim3}', f'UHI_LR{height_lim4}',
                 f'UHI_px_mean', f'UHI_LR{height_lim1}_mean', f'UHI_LR{height_lim2}_mean',
                 f'UHI_LR{height_lim3}_mean', f'UHI_LR{height_lim4}_mean', 'Min_Value_used', 'nbg'
             ]
        }
        encoding.update({
            'lon_bnds': comp,
            'lat_bnds': comp,
            'lon': {'dtype': 'float64'},
            'lat': {'dtype': 'float64'}
        })


        dataset_uhi = xr.Dataset({
            'UHI_px': (['time', 'y', 'x'], uhi_data1),
            f'UHI_LR{height_lim1}': (['time', 'y', 'x'], uhi_data_elv1),
            f'UHI_LR{height_lim2}': (['time', 'y', 'x'], uhi_data_elv2),
            f'UHI_LR{height_lim3}': (['time', 'y', 'x'], uhi_data_elv3),
            f'UHI_LR{height_lim4}': (['time', 'y', 'x'], uhi_data_elv4),
            f'UHI_px_mean': (['time', 'y', 'x'], rural_mean_data),
            f'UHI_LR{height_lim1}_mean': (['time', 'y', 'x'], rural_mean_data_elv1),
            f'UHI_LR{height_lim2}_mean': (['time', 'y', 'x'], rural_mean_data_elv2),
            f'UHI_LR{height_lim3}_mean': (['time', 'y', 'x'], rural_mean_data_elv3),
            f'UHI_LR{height_lim4}_mean': (['time', 'y', 'x'], rural_mean_data_elv4),
            'Min_Value_used': (['time', 'y', 'x'], Min_Value_values),
            'nbg': (['time', 'y', 'x'], current_nbg_values),

        },
            coords={'time': temperature_data['time'],
                    'lon': (['y', 'x'], lon),
                    'lat': (['y', 'x'], lat),
                    'corner': [0, 1, 2, 3],
                    'height': (['height'], height)
                   },
            attrs=uhi_attrs)
        # Assign bounds with explicit dimensions
        dataset_uhi['lon_bnds'] = (('y', 'x', 'corner'), lon_bnds)
        dataset_uhi['lat_bnds'] = (('y', 'x', 'corner'), lat_bnds)

        # Add CF-compliant metadata
        dataset_uhi['lon'].attrs['bounds'] = 'lon_bnds'
        dataset_uhi['lat'].attrs['bounds'] = 'lat_bnds'

        # Set output filename
        print('OOKKKKKKKKK')
        output_filename = f'Urban_Heat_Island_data_{filename[4:]}'
        dataset_uhi.to_netcdf(os.path.join(output_path, output_filename), encoding=encoding)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

   
