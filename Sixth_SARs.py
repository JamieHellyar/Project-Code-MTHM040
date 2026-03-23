# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:48:41 2026

@author: Jamie
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
import scipy
import iris
from scipy.spatial import cKDTree

# First I will load the model data for 333 m resolution so I can define the high-variability areas
ROOT = '/Users/Jamie/Documents/Wind Project/Model Data/'
filepath_333m = ROOT+'333m/{}_333m_wind_speed_12hr.nc'

# For the 28th of June
date = '20250627T1800'

# I will load the data for each resolution using iris
data_333m = iris.load_cube(filepath_333m.format(date))

# For all the wind speeds
ws_all = np.array(data_333m.data)
ws_std = np.std(ws_all, axis=0)	# To define their standard deviations at each pixel

# I can now define a threshold for high variability
threshold = np.percentile(ws_std, 80)	# For the top 20 percentile of variability over the days
high_var_mask = ws_std > threshold	# I will mask out data points above this threshold
rest = ~high_var_mask		# Now the rest of the data will only be lower variability regions


# Path to home directory (in my case, an external storage drive)
HOME_DIR = '/Users/Jamie/Documents/Wind Project/'

# Load MOD333 (reflectance + geometry together)
mod = xr.open_dataset(HOME_DIR + 'MOD333_2025-06-28_0815.nc')

# Again to define lon, lat and reflectance
lat = mod['lat'].values
lon = mod['lon'].values
R = mod['band2_reflectance'].values

# And now for the gometry 
sza = mod['solar_zenith'].values
saa = mod['solar_azimuth'].values
vza = mod['satellite_zenith'].values
vaa = mod['satellite_azimuth'].values


#%%

# Again to use the function to find sun glint angles
def GlintAngle(solar_zenith, solar_azimuth,
                sat_zenith, sat_azimuth):

    d = np.deg2rad(abs(sat_azimuth - solar_azimuth))
    a = np.cos(np.deg2rad(sat_zenith)) * np.cos(np.deg2rad(solar_zenith))
    b = np.sin(np.deg2rad(sat_zenith)) * np.sin(np.deg2rad(solar_zenith)) * np.cos(d)

    glint_radiance = np.arccos(a - b)

    return np.rad2deg(glint_radiance)

# To call on the function
SGA = GlintAngle(sza, saa, vza, vaa)

#%%

# Now to define the digtised curves
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
#%%
 
# Now first for the 1D model:
 
# FOr the wind speeds corresponding to the digitised values
wind_speeds = np.arange(0, 16)  # 0 to 15 m/s
 
# Now I can interpolate each Cox–Munk curve onto a common zenith grid
theta_common = np.linspace(-20, 70, 300)   # For the theta range
ref_curves = {}
for w in wind_speeds:
    x = np.array(dig[w]["x"])
    y = np.array(dig[w]["y"])
    # I need to remove duplicates / sort
    order = np.argsort(x)
    x, y = x[order], y[order]
    # Then to interpolate
    ref_curves[w] = np.interp(theta_common, x, y,
                              left=np.nan, right=np.nan)


# Now as I need to consider the geometry, I need to factor in the peak of each curve

theta_peak = {} 	# For the peak theta value

for w in wind_speeds:	# For each wind speed
    y = ref_curves[w]	# Set the reflectance curves as the y axis
    idx_peak = np.nanargmax(y)	# To find the index of the greatest reflectance
    theta_peak[w] = theta_common[idx_peak]	# Now to find where the peaks are

theta_rel_curves = {}	# To create the theta reflectance empty list

# Now for each wind speed
for w in wind_speeds:	
    theta_rel_curves[w] = theta_common - theta_peak[w]	# Define the new curves centred with the peak at 0
    





#%%

# Now there is no resolution issues
# For the reflectance + geometry grid
lat_flat = mod['lat'].values.ravel()
lon_flat = mod['lon'].values.ravel()

# I need to flatten the current data
R_flat   = R.ravel()       # Taking reflectance data
vza_flat = vza.ravel()     # The satellite zenith
SGA_flat = SGA.ravel()     # And the sun glint angle


# Now to define the latitude constraint for Crete region
lat_min = 33.0
lat_max = 36.0




#%%

# I want to mask for only valid pixels
valid = (
    np.isfinite(R_flat) &
    np.isfinite(SGA_flat) &
    np.isfinite(vza_flat) &
    (R_flat > 0) &		 # Only for finite valid R, vza and SGA values
    (SGA_flat < 15) &		 # For the glint-dominated regime, only SGA<15 degrees
    (lat_flat >= lat_min) &		# And in the correct latitude region
    (lat_flat <= lat_max)
)




 
# Now I can redefine the valid variable pixels I will use
lat_use=lat_flat[valid]
lon_use=lon_flat[valid]
theta_use = SGA_flat[valid]
R_use     = R_flat[valid]
vza_use   = vza_flat[valid]

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

    diff = np.abs(model_R - R_obs)		# I can then find the difference between the Cox-Munk reflectance and the observed reflectance

    # Now, for all valid differences
    if not np.all(np.isnan(diff)):
        wind_est[pixel] = wind_speeds[np.nanargmin(diff)]	# I can select the nearest Cpx-Munk curve to the observed reflectance


#%%

# To define the wind map, I need to flatten the array and apply the valid masking
wind_map = np.full(R.shape, np.nan)
wind_map_flat = wind_map.ravel()
wind_map_flat[valid] = wind_est
wind_map = wind_map_flat.reshape(R.shape)	# Then I can reshape back to the wind map grid that matches the reflectance data grid




#%%

# To define the model data longitudes and latitudes
model_lats = data_333m.coord('latitude').points
model_lons = data_333m.coord('longitude').points

# I can now build the full 333 m model grid
model_lon_grid, model_lat_grid = np.meshgrid(model_lons, model_lats)

# I want to flatten the coordinate data
model_lon_flat = model_lon_grid.ravel()
model_lat_flat = model_lat_grid.ravel()

# And also flatten low-variability mask 
low_var_flat = rest.ravel()

# Now I will build KDTree from all the model grid points
model_points = np.column_stack((model_lon_flat, model_lat_flat))
tree_var = cKDTree(model_points)

# I need to find the nearest model cell for each MODIS pixel
modis_points = np.column_stack((lon_use, lat_use))
dist_var, idx_var = tree_var.query(modis_points, k=1)	# For the distance and the index at each point

# Now I will store only MODIS pixels which is nearest the model
mask_near = low_var_flat[idx_var]


#%%

# For the no masking case, for all wind speeds and data points
wind_est_all = wind_est.copy()
lon_all = lon_use.copy()
lat_all = lat_use.copy()


# And for the masking case
wind_est = wind_est[mask_near]
lon_use = lon_use[mask_near]
lat_use = lat_use[mask_near]













#%%

# Now for the SARs data

import os
import xarray as xr

HOME_DIR = r'C:\Users\Jamie\Documents\Wind Project'

# To extract each SARs file I may need
files = [
    "cmems_obs-wind_med_phy_nrt_l3-s1a-sar-asc-0.01deg_P1D-i_1765813699917.nc",
    "SAR_2025-06-28.nc",
    "SAR333_2025-06-28.nc"
]

import numpy as np
import matplotlib.pyplot as plt


file_path = os.path.join(HOME_DIR, "SAR333_2025-06-28.nc")
sar = xr.open_dataset(file_path)	# To open the SAR333 file for the 333m resolution data

# To define the variables
wind = sar['wind_speed'].isel(time=0)   
lat = sar['lat']
lon = sar['lon']


# To define the souhtern coast of Crete bounds
lat_min, lat_max = 33.0, 36.0
lon_min, lon_max = 23.0, 27.0

# And to create a mask for this region
region_mask = (
    (lat >= lat_min) & (lat <= lat_max) &
    (lon >= lon_min) & (lon <= lon_max)
)

# Now to apply the mask
wind_sar = wind.where(region_mask)

#%%

# To make the comparisons, I need them on the same grid, so first I will flatten SAR
sar_wind_flat = wind_sar.values.ravel()
sar_lat_flat = lat.where(region_mask).values.ravel()
sar_lon_flat = lon.where(region_mask).values.ravel()

# Now to remove any NaNs
valid_sar = np.isfinite(sar_wind_flat)

# And to apply this to my variables
sar_wind_flat = sar_wind_flat[valid_sar]
sar_lat_flat = sar_lat_flat[valid_sar]
sar_lon_flat = sar_lon_flat[valid_sar]

# Now I can build the KDTree from SAR to put both data sets on the same grid 
sar_points = np.column_stack((sar_lon_flat, sar_lat_flat))
tree_sar = cKDTree(sar_points)	# To define the tree for SARs

# MODIS points
modis_points = np.column_stack((lon_use, lat_use))  # To make the MODIS grid of lon and lat values

# Now to find the distance and index for each point from the MODIS grid using the tree I made
dist, idx = tree_sar.query(modis_points, k=1)

# I can define a distance threshold
mask_collocated = dist < 0.01

# And now to define the wind speeds using this tree masking
modis_wind = wind_est[mask_collocated]
sar_wind = sar_wind_flat[idx[mask_collocated]]


# Also, can consider percentiles to make spatial analysis only, ignoring magnitude

def top_percentile(arr):
    ranks = rankdata(arr, method='average')
    return 100 * (ranks - 1) / (len(arr) - 1)


#%%

# Now I can conduct statiscitcal comparisons based on both the wind speeds estimates


# First I will check the linear fit
slope, intercept = np.polyfit(sar_wind, modis_wind, 1)	# To find the slope and intercept

# Next for the correlation and R^2
r = np.corrcoef(sar_wind, modis_wind)[0, 1]
r2 = r**2

# For the bias & RMSE
diff = modis_wind - sar_wind	# To find the difference between wind speeds for each individual point
bias = np.mean(diff)		# The bias can be found by finding the mean of all the differences
rmse = np.sqrt(np.mean(diff**2))	# And hence I can also obtain the RMSE using the differences


# For the percentiles

# To define the percentile variables
sar_pct = top_percentile(sar_wind)
modis_pct = top_percentile(modis_wind)


slope_pct, intercept_pct = np.polyfit(sar_pct, modis_pct, 1)	# To find the slope and intercept
r_pct = np.corrcoef(sar_pct, modis_pct)[0,1]		# Next for the correlation and R^2
r2_pct = r_pct**2
diff_pct = modis_pct - sar_pct			# Again to find the difference
bias_pct = np.mean(diff_pct)			# The bias can be found by finding the mean of all the differences
rmse_pct = np.sqrt(np.mean(diff_pct**2))		# And hence I can also obtain the RMSE using the differences


#%%

# And for the no masking case:
    
# To collocate all of the pixels
modis_points_all = np.column_stack((lon_all, lat_all))
dist_all, idx_all = tree_sar.query(modis_points_all, k=1)

# Again to define a collocation threshold
mask_collocation_all = dist_all < 0.01

# Again to define the variables
modis_wind_all = wind_est_all[mask_collocation_all]
sar_wind_all = sar_wind_flat[idx_all[mask_collocation_all]]

# Now to calculate the statistics
slope_all, intercept_all = np.polyfit(sar_wind_all, modis_wind_all, 1)	# To find the slope and intercept
r_all = np.corrcoef(sar_wind_all, modis_wind_all)[0,1]
r2_all = r_all**2
bias_all = np.mean(modis_wind_all - sar_wind_all)
rmse_all = np.sqrt(np.mean((modis_wind_all - sar_wind_all)**2))

# For the percentiles:
    
sar_pct_all = top_percentile(sar_wind_all)
modis_pct_all = top_percentile(modis_wind_all)


slope_pct_all, intercept_pct_all = np.polyfit(sar_pct_all, modis_pct_all, 1)
r_pct_all = np.corrcoef(sar_pct_all, modis_pct_all)[0,1]
r2_pct_all = r_pct_all**2
diff_pct_all = modis_pct_all - sar_pct_all
bias_pct_all = np.mean(diff_pct_all)
rmse_pct_all = np.sqrt(np.mean(diff_pct_all**2))






#%%

# To print the results:
    
print("ALL pixels:")
print("R^2:", r2_all)
print("Bias:", bias_all)
print("RMSE:", rmse_all)
print("R^2 (percentile):", r2_pct_all)
print("Slope (percentile):", slope_pct_all)

print("\nLOW variability pixels:")
print("R^2:", r2)
print("Bias:", bias)
print("RMSE:", rmse)
print("Slope :", slope)
print("R^2 (percentile):", r2_pct)
print("Slope (percentile):", slope_pct)

print("N all:", len(modis_wind_all))
print("N low-var:", len(modis_wind))

print("Fraction retained:", np.sum(mask_near) / len(mask_near))	# To see how many pixels remained after masking the high variability regions




#%%

# As advised, I now want to consider a density to my scatter graph to make it easier to interpret

plt.figure(figsize=(8,6))

# To define the histogram to visually show the correlations
plt.hist2d(
    sar_wind,
    modis_wind,
    bins=17,
    range=[[0,15],[0,15]],
    cmap='inferno'
)

plt.plot([0,15],[0,15],'w--',lw=1) # I will plot for 0-15 m/s for both as the MODIS version is limited to this range
    
x = np.linspace(0,15,100)
plt.plot(x, x+1, 'w:', lw=1)
plt.plot(x, x-1, 'w:', lw=1)

# For the labels
plt.xlabel('SAR Wind Speed (m/s)',fontsize=18)
plt.ylabel('MODIS Wind Speed Estimates (m/s)',fontsize=18)

# To define the colourbar
cb = plt.colorbar()
cb.set_label("Number of pixels", fontsize=16)

plt.xlim(0,15)
plt.ylim(0,15)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()
























