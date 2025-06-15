import xarray as xr

# Function to classify grid points as urban, rural, water, or sea
def classify_grid_points(args):
    town_fract, nature_fract, sea_fract, water_fract, urban_threshold, rural_threshold, sea_water_threshold = args

    urban_mask = town_fract > urban_threshold
    rural_mask = nature_fract > rural_threshold
    water_mask = water_fract > sea_water_threshold
    sea_mask = sea_fract > sea_water_threshold

    return xr.where(urban_mask, 'urban',
                    xr.where(rural_mask, 'rural',
                             xr.where(water_mask, 'water',
                                      xr.where(sea_mask, 'sea', 'other')))) ## 'other' is put as default value

