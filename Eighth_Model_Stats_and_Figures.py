# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:53:32 2026

@author: Jamie
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
import iris
from datetime import datetime



HOME_DIR = '/Users/Jamie/Documents/Wind Project/'

# Can do the same for MOD333_2025-06-23_0815.nc and MOD333_2025-07-03_0815.nc
mod = xr.open_dataset(HOME_DIR + 'MOD333_2025-06-28_0815.nc')

# To define the latitude and longitude variables
lat = mod['lat'].values
lon = mod['lon'].values

# To define the variables
R   = mod['band2_reflectance'].values
sza = mod['solar_zenith'].values
saa = mod['solar_azimuth'].values
vza = mod['satellite_zenith'].values
vaa = mod['satellite_azimuth'].values


#%%
# Again to calculate the sun-glint angles
def GlintAngle(solar_zenith, solar_azimuth,
               sat_zenith, sat_azimuth):

    d = np.deg2rad(abs(sat_azimuth - solar_azimuth)) # To convert from degrees to radians

    a = np.cos(np.deg2rad(sat_zenith)) * np.cos(np.deg2rad(solar_zenith))
    b = np.sin(np.deg2rad(sat_zenith)) * np.sin(np.deg2rad(solar_zenith)) * np.cos(d)

    return np.rad2deg(np.arccos(a - b))		# To convert back from radians to degrees

SGA = GlintAngle(sza, saa, vza, vaa)


#%%

# For digitising the Cox-Munk curves again

dig = {
    0: {
        "x": [3.0, 4.68, 6.12, 7.56, 9.24, 11.16, 13.08, 15.72, 18.12, 20.28,
              23.16, 25.32, 27.24, 29.88, 32.28, 33.72, 35.64, 36.6, 38.04, 39.24,
              41.16, 43.32, 45.72],
        "y": [0.011799, 0.038399, 0.076199, 0.1126, 0.1588, 0.2176, 0.2834,
              0.3422, 0.401, 0.4388, 0.4738, 0.4773, 0.4682, 0.4262, 0.3814, 0.3338,
              0.289, 0.2484, 0.2008, 0.1532, 0.1126, 0.062199, 0.025799]
    },

    1: {
        "x": [2.628, 4.5392, 6.2116, 7.884, 9.3174, 10.512, 11.706, 13.379,
              14.573, 15.529, 17.44, 18.635, 20.307, 22.457, 24.369, 25.563,
              27.713, 29.386, 31.058, 32.253, 33.686, 35.119, 35.836,
              36.792, 37.747, 38.703, 39.659, 41.092, 42.287, 43.242,
              43.959, 44.915, 46.348, 48.259],
        "y": [0.0014205, 0.022727, 0.056818, 0.09375, 0.12358, 0.15341,
              0.19318, 0.22301, 0.25426, 0.27983, 0.31108, 0.33523,
              0.36364, 0.39062, 0.40341, 0.40341, 0.39631, 0.37784, 0.35227,
              0.32812, 0.29688, 0.27131, 0.25142, 0.22301, 0.20028,
              0.1733, 0.14489, 0.12074, 0.096591, 0.073864, 0.055398,
              0.036932, 0.018466, 0.0028409]
    },

    2: {
        "x": [2.628, 4.7782, 6.6894, 9.0785, 10.99, 12.423, 14.334, 16.485,
              18.157, 20.068, 22.218, 24.846, 25.546, 27.952, 30.819, 33.208,
              34.642, 35.836, 37.509, 38.703, 40.137, 41.331, 43.003,
              45.631, 47.782, 50.41, 53.754, 57.816, 60.922],
        "y": [0.0014205, 0.02983, 0.056818, 0.09375, 0.125, 0.15483,
              0.18608, 0.21875, 0.24148, 0.26705, 0.28693, 0.30256, 0.30114,
              0.29545, 0.27415, 0.24574, 0.22159, 0.19602, 0.17472,
              0.15199, 0.12642, 0.10511, 0.079545, 0.052557, 0.03267,
              0.018466, 0.011364, 0.014205, 0.024148]
    },

    3: {
        "x": [-0.23891, 3.5836, 6.2116, 8.3618, 10.99, 14.096, 18.157,
              22.696, 27.474, 31.297, 35.597, 38.225, 40.853, 43.959,
              50.171, 55.427, 62.116],
        "y": [-0.0014245, 0.019943, 0.045584, 0.078348, 0.10541, 0.14672,
              0.19231, 0.2265, 0.23504, 0.2151, 0.17806, 0.14387,
              0.11396, 0.076923, 0.039886, 0.022792, 0.027066]
    },

    4: {
        "x": [-1.9113, 3.5836, 7.4061, 11.706, 15.768, 20.785, 25.802,
              32.014, 36.314, 40.614, 43.959, 51.604, 57.338],
        "y": [0.0042735, 0.029915, 0.059829, 0.098291, 0.13818, 0.17521,
              0.19658, 0.18234, 0.15242, 0.11681, 0.084046, 0.048433,
              0.035613]
    },

    5: {
        "x": [-6.9283, -0.23891, 5.0171, 10.751, 14.812, 19.829, 25.324,
              31.058, 36.314, 41.809, 47.065, 52.082, 59.727, 65.7],
        "y": [0.0042735, 0.018519, 0.039886, 0.082621, 0.11396, 0.14387,
              0.16667, 0.16239, 0.1396, 0.10826, 0.07265, 0.052707,
              0.034188, 0.035613]
    },

    6: {
        "x": [-5.4949, 2.1502, 8.1229, 12.423, 17.679, 22.935, 29.147,
              34.881, 41.092, 45.87, 52.321, 59.01, 66.655],
        "y": [0.011396, 0.032764, 0.059829, 0.089744, 0.11966, 0.13818,
              0.14815, 0.13675, 0.11111, 0.082621, 0.059829, 0.038462,
              0.037037]
    },

    7: {
        "x": [-12.184, -8.6007, -4.0614, -0.23891, 3.3447, 6.6894, 9.7952,
              12.662, 15.29, 18.396, 21.024, 24.13, 26.758, 29.147,
              32.253, 35.836, 38.225, 40.853, 43.003, 44.915, 47.543,
              49.932, 52.799, 57.099, 60.922, 63.311, 68.328],
        "y": [0.012784, 0.015625, 0.015625, 0.026989, 0.039773, 0.055398,
              0.068182, 0.083807, 0.098011, 0.10938, 0.12074, 0.12642,
              0.13068, 0.1321, 0.12926, 0.12216, 0.11364, 0.10227,
              0.09233, 0.083807, 0.075284, 0.065341, 0.058239,
              0.049716, 0.044034, 0.041193, 0.038352]
    },

    8: {
        "x": [-17.679, -12.901, -8.6007, -5.256, -0.23891, 2.8669,
              5.7338, 7.884, 10.751, 13.618, 16.962, 19.59, 21.98,
              24.13, 26.758, 29.147, 32.491, 35.358, 37.986, 40.375,
              44.198, 46.587, 48.737, 51.126, 53.993, 57.099,
              60.444, 64.027, 67.133],
        "y": [0.014205, 0.014205, 0.018466, 0.022727, 0.03125, 0.036932,
              0.046875, 0.06108, 0.071023, 0.082386, 0.09375, 0.10369,
              0.1108, 0.11506, 0.11932, 0.11932, 0.11648, 0.11364,
              0.10938, 0.10369, 0.09233, 0.082386, 0.071023, 0.065341,
              0.059659, 0.053977, 0.048295, 0.045455, 0.044034]
    },



    9: {
        "x": [-12.18, -7.65, -4.54, -0.96, 3.34, 6.21, 8.84, 12.42, 16.25,
              20.31, 23.41, 26.28, 28.91, 31.54, 34.40, 37.51, 41.09,
              43.96, 46.83, 50.41, 53.28, 56.38, 59.73, 63.55],        
        "y": [0.0185, 0.0241, 0.0284, 0.0341, 0.0412, 0.0511, 0.0639,
              0.0724, 0.0852, 0.0966, 0.1037, 0.1080, 0.1094, 0.1108,
              0.1094, 0.1037, 0.0980, 0.0909, 0.0838, 0.0724, 0.0668,
              0.0611, 0.0554, 0.0526]
    },

    10: {
        "x": [-13.38, -8.36, -3.34, 0.96, 4.54, 8.12, 11.95, 16.01,
              19.59, 23.41, 26.04, 28.19, 31.30, 34.64, 38.23,
              42.53, 46.83, 50.41, 53.99, 58.29, 62.12],
        "y": [0.0213, 0.0241, 0.0313, 0.0384, 0.0483, 0.0568, 0.0668,
              0.0810, 0.0881, 0.0966, 0.1009, 0.1023, 0.1023,
              0.1009, 0.0966, 0.0909, 0.0824, 0.0724, 0.0653,
              0.0597, 0.0568]
    },

    11: {
        "x": [-10.03, -5.49, -0.96, 3.34, 6.69, 10.51, 14.10, 17.44,
              20.31, 22.94, 26.52, 29.62, 32.49, 35.60, 38.46,
              41.57, 45.39, 48.98, 53.28, 56.38, 58.77, 62.12,
              64.98, 67.37],
        "y": [0.0256, 0.0298, 0.0369, 0.0426, 0.0540, 0.0639,
              0.0724, 0.0781, 0.0838, 0.0881, 0.0923, 0.0966,
              0.0966, 0.0938, 0.0923, 0.0909, 0.0824, 0.0753,
              0.0710, 0.0653, 0.0639, 0.0625, 0.0597, 0.0597]
    },

    12: {
        "x": [-9.08, -3.82, 1.43, 5.97, 10.03, 14.10, 17.68, 21.50,
              25.09, 28.91, 32.25, 35.12, 38.46, 41.33, 44.20,
              46.59, 48.98, 52.08, 55.67, 58.77, 62.35, 66.42],
        "y": [0.0270, 0.0341, 0.0398, 0.0483, 0.0611, 0.0710,
              0.0781, 0.0838, 0.0881, 0.0909, 0.0923, 0.0909,
              0.0909, 0.0881, 0.0852, 0.0810, 0.0767, 0.0724,
              0.0668, 0.0639, 0.0611, 0.0611]
    },

    13: {
        "x": [-17.68, -13.86, -10.27, -5.73, -1.67, 1.91, 5.73,
              9.56, 13.62, 18.16, 22.46, 26.04, 30.10, 33.69,
              36.79, 39.90, 43.24, 46.59, 52.08, 56.38, 60.68,
              64.51, 69.28],
        "y": [0.0227, 0.0256, 0.0284, 0.0327, 0.0384, 0.0440,
              0.0511, 0.0597, 0.0668, 0.0739, 0.0810, 0.0866,
              0.0895, 0.0895, 0.0895, 0.0881, 0.0852, 0.0810,
              0.0753, 0.0710, 0.0696, 0.0668, 0.0653]
    },

    14: {
        "x": [-20.78, -15.77, -11.23, -6.93, -1.19, 3.82, 7.88,
              11.95, 16.72, 21.02, 24.61, 28.19, 32.73, 37.27,
              41.57, 45.39, 51.84, 57.58, 61.88, 65.46, 68.57],
        "y": [0.0227, 0.0270, 0.0313, 0.0341, 0.0398, 0.0483,
              0.0540, 0.0625, 0.0696, 0.0767, 0.0810, 0.0838,
              0.0852, 0.0852, 0.0838, 0.0838, 0.0781, 0.0710,
              0.0696, 0.0696, 0.0696]
    },

    15: {
        "x": [-16.01, -10.03, -5.26, -0.24, 5.02, 10.99, 15.53,
              20.07, 24.37, 29.15, 32.97, 37.27, 40.61, 45.63,
              49.22, 53.99, 59.01, 63.07],
        "y": [0.0313, 0.0355, 0.0398, 0.0455, 0.0497, 0.0582,
              0.0639, 0.0710, 0.0781, 0.0810, 0.0838, 0.0824,
              0.0824, 0.0824, 0.0795, 0.0753, 0.0739, 0.0724]
    }
    
}



wind_speeds = np.arange(0, 16)		# To create an array for wind speeds 0 to 15m/s
theta_common = np.linspace(-20, 70, 300)	# Over the theta range of -20 to 70 degrees

ref_curves = {}		# To create an empty list for the interpolated reflectance curves

for w in wind_speeds:			# For each wind speed
    x = np.array(dig[w]["x"])			# To extract the angle and reflectance data
    y = np.array(dig[w]["y"])

    order = np.argsort(x)			# To order the data in terms of theta angle
    x, y = x[order], y[order]

    # Now I can interpolate the data onto a common grid
    ref_curves[w] = np.interp(theta_common, x, y,	
                              left=np.nan, right=np.nan)

theta_peak = {}		# I need to create an empty list for the peak theta values

for w in wind_speeds:			# For each wind speed
    idx_peak = np.nanargmax(ref_curves[w])	# I can find the index for where the maximum curve is
    theta_peak[w] = theta_common[idx_peak]	# I can now set the peak theta values at this index

theta_rel_curves = {}		# Now for the theta reflectance curves array
for w in wind_speeds:			# For each wind speed
    theta_rel_curves[w] = theta_common - theta_peak[w]	# I can now shift each curves position so the peak is at 0



#%%

# Now I can convert all my data into 1D arrays
lat_flat = lat.ravel()
lon_flat = lon.ravel()
R_flat   = R.ravel()
SGA_flat = SGA.ravel()
vza_flat = vza.ravel()

# Now I want to create a mask for only valid data as before
valid = (
    np.isfinite(R_flat) & 	# For only finite values
    np.isfinite(SGA_flat) &
    (R_flat > 0) &	# Reflectance greater than 1
    (SGA_flat < 15) &	# SGA less than 15
    (lat_flat >= 34) & (lat_flat <= 36) &	# For the desired lat and lon range
    (lon_flat >= 23) & (lon_flat <= 27)
)

# To apply my mask
lat_use   = lat_flat[valid]
lon_use   = lon_flat[valid]
theta_use = SGA_flat[valid]
R_use     = R_flat[valid]





#%%

# I need to stack the reflectance curves into vertical arrays
model_curves = np.vstack([ref_curves[w] for w in wind_speeds])
theta_rel    = np.vstack([theta_rel_curves[w] for w in wind_speeds])

valid_pix = len(R_use)		# For the valid pixels
wind_est = np.full(valid_pix, np.nan)	# The estimate wind speeds for each pixel

# Now I can loop over each individual pixel
for pixel in range(valid_pix):	
    theta = theta_use[pixel]		# To record the SGA for each pixel
    R_obs = R_use[pixel]		# And the reflectance for each

    # I can now compare against all of the model wind speeds
    model_R = np.array([
        np.interp(theta, theta_rel[j], model_curves[j])	# For the model curve, I’ll use the shifted theta and interpolate it to my pixels theta value
        for j in range(len(wind_speeds))
    ])

    diff = np.abs(model_R - R_obs)		# I can then find the difference between the Cox-Munk reflectance and the observed reflectancediff 

    # Now, for all valid differences
    if not np.all(np.isnan(diff)):
        wind_est[pixel] = wind_speeds[np.nanargmin(diff)]	# I can select the nearest Cox-Munk curve to the observed reflectance





#%%

# To remove any 0 values from the inversion
valid_modis = np.isfinite(wind_est)

lat_modis  = lat_use[valid_modis]	# To apply the mask to the lat lon and wind
lon_modis  = lon_use[valid_modis]
wind_modis = wind_est[valid_modis]







#%%

# Now I want to load the UKESM model datasets
ROOT = '/Users/Jamie/Documents/Wind Project/Model Data/'
filepath_333m = ROOT+'333m/{}_333m_wind_speed_12hr.nc'
filepath_1km = ROOT+'1km/{}_1km_wind_speed_12hr.nc'
filepath_global = ROOT+'global/{}_global_wind_speed_12hr.nc'

# I need to select the main test day I want to look at
date = '20250627T1800'  # (I will  repeat the exact same code for the 23rd of June and 3rd of July)

# To load these datasets:
data_333m = iris.load_cube(filepath_333m.format(date))
data_1km  = iris.load_cube(filepath_1km.format(date))
data_10km = iris.load_cube(filepath_global.format(date))

# I need to convert the global data to -180 to 180
lon = data_10km.coord('longitude')
lon_wrap = ((lon.points+180)%360)-180 # To conver the coordinates

sorted_idx = np.argsort(lon_wrap) # I need to sort the data for longitudes
lon.points = lon_wrap[sorted_idx] # And now I can redefine my longitudes

lon_axis = data_10km.coord_dims(lon)[0] # To find axis that corresponds to longitude data
data_10km.data=np.take(data_10km.data,sorted_idx,axis=lon_axis) # Reordering teh data to match the reordered longitude coordinate
lon.circular=True






#%%

# Now I want to extract at the correct time for each dataset:
    
time_coord = data_1km.coord('time')
times = time_coord.units.num2date(time_coord.points)

#First want to look for a time around 08:15 as that was the time of the MODIS overpass

from datetime import datetime
import numpy as np

# To define the target time
target_time = datetime(2025, 6, 28, 8, 15)

# Now I can look at the difference in each time from the target times
time_diff = np.array([abs((t - target_time).total_seconds()) for t in times])


# And now to find the smallest time difference
i = np.argmin(time_diff)

# And now I can print the closest model time
print("Closest model time:", times[i])

# This gives 08:11
# Now I have index i that I can use to make the plots


#%%

# And now to extract the data at 08:11:00

# For the wind speed resoluutions
ws_333m = np.array(data_333m.data[i,:,:])
ws_1km  = np.array(data_1km.data[i,:,:])
ws_10km = np.array(data_10km.data[i,:,:])

# And for the coordinates at each resolution
lat_333m = data_333m.coord('latitude').points
lon_333m = data_333m.coord('longitude').points

lat_1km = data_1km.coord('latitude').points
lon_1km = data_1km.coord('longitude').points

lat_10km = data_10km.coord('latitude').points
lon_10km = data_10km.coord('longitude').points






#%%

# Now I can define the range of latitude and longitude values I will work with
lat_mask = (lat_333m >= 34) & (lat_333m <= 36)
lon_mask = (lon_333m >= 23) & (lon_333m <= 27)

# Now to apply the mask at the 333 m resolution
ws333 = ws_333m[np.ix_(lat_mask, lon_mask)]
lat333 = lat_333m[lat_mask]
lon333 = lon_333m[lon_mask]

# And now to repeat for the 1 km simulations
lat_mask2 = (lat_1km >= 34) & (lat_1km <= 36)
lon_mask2 = (lon_1km >= 23) & (lon_1km <= 27)

ws1 = ws_1km[np.ix_(lat_mask2, lon_mask2)]
lat1 = lat_1km[lat_mask2]
lon1 = lon_1km[lon_mask2]

# And then finally again for the 10 km resolutions
lat_mask3 = (lat_10km >= 34) & (lat_10km <= 36)
lon_mask3 = (lon_10km >= 23) & (lon_10km <= 27)

ws10 = ws_10km[np.ix_(lat_mask3, lon_mask3)]
lat10 = lat_10km[lat_mask3]
lon10 = lon_10km[lon_mask3]



#%%

# To assess only the spatial distributions, I can look at percentiles

from scipy.stats import rankdata

# To create a function that finds the top percentiles
def top_percentile(array):
    rank = rankdata(arr, method='average')		# Ranking the data by averages
    percentiles = 100 * (rank - 1) / (len(array) - 1)		# Caclulating the percentiles
    return percentiles





#%%

# Now I can create a function for the different statisitcal tests 
# at the 3 resolutions as follows:

# I will define a function that puts the points on the same grid and hence 
# calculating various statistics

# Note: I need to put the collocation in the same grid so that I can run the same function for different resolutions

def compare_resolution(ws_field, lat_grid, lon_grid, label):
    
    # Flatten model grid
    model_wind_flat = ws_field.ravel()
    model_lat_flat  = np.repeat(lat_grid, len(lon_grid))
    model_lon_flat  = np.tile(lon_grid, len(lat_grid))
    
    valid_model = np.isfinite(model_wind_flat)	# For only finite values
    
    # Now, to pair the coordinates
    model_points = np.column_stack((
        model_lon_flat[valid_model],
        model_lat_flat[valid_model]
    ))
    
    # And then to match the wind speeds to this
    model_wind_use = model_wind_flat[valid_model]
    
    # To then build the KDTree
    tree_model = cKDTree(model_points)
    

    # I can now match the MODIS points to the latitudes and longitudes I want to use
    modis_points = np.column_stack((lon_use, lat_use))
    dist, idx = tree_model.query(modis_points, k=1) # To find the distance to the nearest model point and its index
    
    # I can now define the model wind speed at each MODIS pixel
    model_collocated = model_wind_use[idx]
    
    # To ensure each wind speed set contains only finite values
    valid_pair = np.isfinite(wind_est) & np.isfinite(model_collocated)
    
    # To apply this validity mask
    modis_wind = wind_est[valid_pair]
    model_wind = model_collocated[valid_pair]

    # To account for a top down reflectance version:
    reflectance = R_use[valid_pair]
    
    # And then for the percentiles
    modis_pct = top_percentile(modis_wind)
    model_pct = top_percentile(model_wind)
    
    # Now to make the statistics calculations

    # For the correlation and R^2
    r = np.corrcoef(model_wind, modis_wind)[0,1]	# For the correlation and R^2
    r2 = r**2
    
    # For the bias & RMSE
    diff = modis_wind - model_wind	# To find the difference between wind speeds for each individual point
    bias = np.mean(diff)		# The bias can be found by finding the mean of all the differences
    rmse = np.sqrt(np.mean(diff**2))	# And hence I can also obtain the RMSE using the differences

    # I can use np.polyfit again to find the slope and intercept
    slope, intercept = np.polyfit(model_wind, modis_wind, 1)
    
    # And now for the percentiles statistics
    r_pct = np.corrcoef(model_pct, modis_pct)[0,1]
    r2_pct = r_pct**2
    diff_pct = model_pct - modis_pct
    bias_pct = np.mean(diff_pct)
    rmse_pct = np.sqrt(np.mean(diff_pct**2))
    slope_pct, intercept_pct = np.polyfit(model_pct, modis_pct, 1)


    # And now to print all the results
    print(f"N: {len(modis_wind)}")
    print(f"R^2: {r2:.2f}")
    print(f"Bias: {bias:.2f} m/s")
    print(f"RMSE: {rmse:.2f} m/s")
    print(f"Slope: {slope:.2f}")
    print(f"Intercept: {intercept:.2f}")
    print("Percentile comparison:")
    print(f"R^2 (pct): {r2_pct:.2f}")
    print(f"Slope (pct): {slope_pct:.2f}")
    print(f"Bias (pct): {bias_pct:.2f}")
    print(f"RMSE (pct): {rmse_pct:.2f}")
    
    return model_wind, modis_wind



#%%

# To run the function for each resolution
    
model333, modis333 = compare_resolution(ws333, lat333, lon333, "333 m")
model1,   modis1   = compare_resolution(ws1, lat1, lon1, "1 km")
model10,  modis10  = compare_resolution(ws10, lat10, lon10, "10 km")



#%%

# Now finally to plot the 2D histograms;

# I can create a function to run for each resolution
def plot_hist(model, modis, label):
    
    plt.figure(figsize=(6,6))
    

    # To make the histogram
    plt.hist2d(model, modis, bins=17, range=[[0,15],[0,15]],  cmap='inferno')
    plt.plot([0,15],[0,15],'w--',lw=1)
    
    # To add lines for x=y and threshold lines for +-1m/s
    x = np.linspace(0,15,100)
    plt.plot(x, x+1, 'w:', lw=1)
    plt.plot(x, x-1, 'w:', lw=1)
    
    # For the plot labels
    plt.xlabel(f"{label} Model wind (m/s)", fontsize=18)
    plt.ylabel("MODIS wind (m/s)", fontsize=18)

    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # To create the colourbar
    cb = plt.colorbar()
    cb.set_label("Number of pixels", fontsize=16)
    
    plt.tight_layout()
    plt.show()


#%%

# And now to call on the function to plot the different histograms
plot_hist(model333, modis333, "333 m")
plot_hist(model1,   modis1,   "1 km")
plot_hist(model10,  modis10,  "10 km")

















#%%

# To compare differences over different sun glint angles
sga_bins = np.arange(0, 11, 1)  # To look at 0-10 degrees

# To loop over different SGA ranges
for i in range(len(sga_bins)-1):
    # To create pairs of wind speeds 1m/s apart, ie 0-1, 1-2, 2-3 etc
    sga_min = sga_bins[i]     
    sga_max = sga_bins[i+1]
    
    # Now I can create a mask to find data points in each bin
    mask = (theta_use >= sga_min) & (theta_use < sga_max)

    # And now to apply this mask to make subsets for each bin
    modis_subset = wind_est[mask]
    lat_subset   = lat_use[mask]
    lon_subset   = lon_use[mask]

    # Now to match these points to the model
    modis_points = np.column_stack((lon_subset, lat_subset))

    # To flatten 333m model grid into a list of values
    model_wind_flat = ws333.ravel()
    model_lat_flat  = np.repeat(lat333, len(lon333))
    model_lon_flat  = np.tile(lon333, len(lat333))
    # And removing any invalid values
    valid_model = np.isfinite(model_wind_flat)
 
    # Now I can build coordinate pairs between the model and my MODIS estimates
    model_points = np.column_stack((
        model_lon_flat[valid_model],
        model_lat_flat[valid_model]
    ))
    # For the valid data points
    model_wind_use = model_wind_flat[valid_model]
 
    # Again I need to construct a KDTree
    tree = cKDTree(model_points)

    # To find the distance to the nearest model point and its index
    dist, idx = tree.query(modis_points, k=1) 	

    # I can now define the model wind speed at each MODIS pixel
    model_collocated = model_wind_use[idx]	

    # Now to make a mask for only valid MODIS and model win speeds
    valid_pair = np.isfinite(modis_subset) & np.isfinite(model_collocated)
   
    # Now I will apply the mask to both subsets
    modis = modis_subset[valid_pair]
    model = model_collocated[valid_pair]

    # To calculate the statistics for each range
    r = np.corrcoef(model, modis)[0,1]
    r2 = r**2
    rmse = np.sqrt(np.mean((modis-model)**2))



#%%

# For top down reflectance version:
    
# I can create a function to be run for each resolution
def compare_reflectance(ws_field, lat_grid, lon_grid, label):

    # First I need to flatten the wind speeds and coordinates into lists of values
    model_wind_flat = ws_field.ravel()
    model_lat_flat  = np.repeat(lat_grid, len(lon_grid))
    model_lon_flat  = np.tile(lon_grid, len(lat_grid))

    valid_model = np.isfinite(model_wind_flat)	# For only finite values

    # Now I can rebuild the pairs of coordinates a s a grid of longitudes and latitudes
    model_points = np.column_stack((
        model_lon_flat[valid_model],
        model_lat_flat[valid_model]
    ))

    # And for the wind speeds I will use accounting for only the valid values
    model_wind_use = model_wind_flat[valid_model]

    # Again I need to construct a KDTree
    tree = cKDTree(model_points)

    # To build the pairs of longitudes and latitudes for the MODIS estimates
    modis_points = np.column_stack((lon_use, lat_use))

    # To find the distance to the nearest model point and its index
    dist, idx = tree.query(modis_points, k=1)	

    # I can now define the model wind speed at each MODIS pixel
    model_collocated = model_wind_use[idx]

    # Now to make a mask for only valid MODIS and model win speeds
    valid_pair = np.isfinite(R_use) & np.isfinite(model_collocated)

    # To define the valid reflectance and model wind speeds
    reflectance = R_use[valid_pair]
    model_wind  = model_collocated[valid_pair]

    return model_wind, reflectance

#%%

# Now I can call on the function for each resolution to find the model wind speeds and refltance values
model333_R, refl333 = compare_reflectance(ws333, lat333, lon333, "333 m")
model1_R,   refl1   = compare_reflectance(ws1, lat1, lon1, "1 km")
model10_R,  refl10  = compare_reflectance(ws10, lat10, lon10, "10 km")




# I need to ensure all reflectance values however are between 0 and 1, so I need to apply a mask for each resolution
mask333=(refl333>=0) & (refl333<=1)
refl333 = refl333[mask333]
mask1=(refl1>=0) & (refl1<=1)
refl1 = refl1[mask1]
mask10=(refl10>=0) & (refl10<=1)
refl10 = refl10[mask10]







#%%

# Now I will define a function that can plot the 2D histogram of reflectance and UKESM wind speeds
def plot_reflectance(model, refl, label):

    plt.figure(figsize=(6,6))

    # For the 2D histogram
    plt.hist2d(model, refl, range=[[0,15],[0,0.5]], bins=40, cmap='inferno') # To make 2D histogram

    # For the labels
    plt.xlabel(f"{label} Model wind (m/s)", fontsize=18)
    plt.ylabel("Reflectance", fontsize=18)

    # To set the colourbar
    cb = plt.colorbar()
    cb.set_label("Pixel count", fontsize=16)

    plt.tight_layout()
    plt.show()

#%%

# I can plot the reflectance for each resolution
plot_reflectance(model333_R, refl333, "333 m")
plot_reflectance(model1_R,   refl1,   "1 km")
plot_reflectance(model10_R,  refl10,  "10 km")


#%%

# To compute the statistics for all 3 resolutions

r = np.corrcoef(model333_R, refl333)[0,1]	# I can find r with the corrcoef function
r2=r**2
print("333m reflectance correlation:", r2)		# To print the results

r = np.corrcoef(model1_R, refl1)[0,1]		# I can find r with the corrcoef function
r2=r**2
print("1km reflectance correlation:", r2)		# To print the results

r = np.corrcoef(model10_R, refl10)[0,1] 		# I can find r with the corrcoef function
r2=r**2
print("10km reflectance correlation:", r2)		# To print the results
