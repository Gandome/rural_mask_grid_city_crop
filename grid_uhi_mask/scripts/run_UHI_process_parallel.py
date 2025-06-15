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

from spatial_UHI_mask.parallel_process import process_folder_parallel
from spatial_UHI_mask.urban_mask import classify_grid_points 



# Loading of Ecoclimap data and parameters definition
ecoclmp_alp = xr.open_dataset("/home/quenumm/Documents/data/PGD/Lea/PGD.2018.nc")
# ecoclmp_alp = xr.open_dataset("/home/cnrm_other/ge/mrmu/quenumm/Recup/data/NFR009b/PGD.2018.nc")
urban_threshold = 0.30
rural_threshold = 0.6
sea_water_threshold = 0.3

town_fract = ecoclmp_alp['SFX.FRAC_TOWN']
nature_fract = ecoclmp_alp['SFX.FRAC_NATURE']
sea_fract = ecoclmp_alp['SFX.FRAC_SEA']
water_fract = ecoclmp_alp['SFX.FRAC_WATER']

grid_classifications = classify_grid_points((town_fract, nature_fract, sea_fract,
                                             water_fract, urban_threshold, rural_threshold, sea_water_threshold))

urban_grid_points = np.argwhere(grid_classifications.values == 'urban')#[14830:16000]
rural_fraction = ecoclmp_alp['SFX.FRAC_NATURE'].rename({'Y': 'x', 'X': 'y'})
urban_fraction = ecoclmp_alp['SFX.FRAC_TOWN'].rename({'Y': 'x', 'X': 'y'})
sea_mask = ecoclmp_alp['SFX.FRAC_SEA'].rename({'Y': 'x', 'X': 'y'})

print(len(urban_grid_points))
# Definition of nO value
nO = 2
nbg = 4
max_iterations = 26
Min_Value=70
# height_lim = 200
height_lim1 = 100
height_lim2 = 150
height_lim3 = 200
height_lim4 = 250

for yr in range(2022, 2023):
    path0 = f"/home/quenumm/Documents/data/data_lea/{yr}/day/test"
    path_rst = f"/home/quenumm/Documents/data/data_lea/{yr}/day/output" # 
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)
    process_folder_parallel(path0, ecoclmp_alp['SFX.ZS'], path_rst, sea_mask, rural_threshold, urban_threshold,
                            rural_fraction, urban_fraction, urban_grid_points, Min_Value, nbg,
                            max_iterations, nO, height_lim1, height_lim2, height_lim3, height_lim4)
