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

from spatial_UHI_mask.calculation_process import process_file

def process_folder_parallel(folder_path, Elevation, output_path, sea_mask, rural_threshold, urban_threshold,
                            nature_fract, town_fract, urban_grid_points, Min_Value, nbg, max_iterations, nO,
                            height_lim1, height_lim2, height_lim3, height_lim4):
    files = sorted(gb.glob(os.path.join(folder_path, '*.nc')))
    with Pool(processes=1) as pool:
        pool.starmap(process_file, [(file_path, Elevation, output_path, sea_mask, rural_threshold, urban_threshold,
                                     nature_fract, town_fract, urban_grid_points, Min_Value, nbg, max_iterations, nO,
                                     height_lim1, height_lim2, height_lim3, height_lim4) for file_path in files])

