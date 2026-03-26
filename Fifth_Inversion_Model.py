# And now for the Inversion work:
 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
import scipy
# Path to my home directory
HOME_DIR = '/Users/Documents/Wind Project/' 
# 28 June reflectance (small regridded file)
REFL_FILE = HOME_DIR + 'MOD02QKM_333_2025-06-28_0815.nc'
# 28 June geometry
GEOM_FILE = HOME_DIR + 'GEOMETRY250_2025-06-28_0815.nc'
 
# To first load the reflectance data
refl_ds = xr.open_dataset(REFL_FILE)
# And then load the geometry data
geom_ds = xr.open_dataset(GEOM_FILE)
 
#%%
 
# To define each variable
lat = refl_ds['lat'].data
lon = refl_ds['lon'].data
 
# I will use band 2 for sun glint calculations
# To define the reflectance and angle variables
R = refl_ds['band2_reflectance'].data
sza = geom_ds['solar_zenith_angle'].data        # Solar zenith angle
saa = geom_ds['solar_azimuth_angle'].data       # Solar azimuth angle
vza = geom_ds['satellite_zenith_angle'].data    # Viewing zenith angle
vaa = geom_ds['satellite_azimuth_angle'].data   # Viewing azimuth angle
 
# To calculate the SGA
def GlintAngle(solar_zenith, solar_azimuth,
                sat_zenith, sat_azimuth):
    d = np.deg2rad(abs(sat_azimuth - solar_azimuth))    # To find the relative azimuth angle and convert it to radians
    a = np.cos(np.deg2rad(sat_zenith)) * np.cos(np.deg2rad(solar_zenith))
    b = np.sin(np.deg2rad(sat_zenith)) * np.sin(np.deg2rad(solar_zenith)) * np.cos(d)
    glint_radiance = np.arccos(a - b)
    return np.rad2deg(glint_radiance)                   # I need to convert it back from radians to degrees
 
SGA = GlintAngle(sza, saa, vza, vaa)
 
#%%

# Now to define all the points I digitised from the Cox-Munk curves again
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
 
# Wind speeds corresponding to the digitised values
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
 
#%%
 
# For now I will start with the 1D inversion
 
# To define the reflectance grid
lat_r = refl_ds['lat'].values.ravel()
lon_r = refl_ds['lon'].values.ravel()
R_flat = R.ravel()
 
# And next the geometry grid
lat_g = geom_ds['lat'].values.ravel()
lon_g = geom_ds['lon'].values.ravel()
vza_flat = vza.ravel()
 
from scipy.spatial import cKDTree
 
# From here I can build a KD-tree in lat/lon space
geom_tree = cKDTree(np.column_stack((lat_g, lon_g)))
 
# For each reflectance pixel, I'll find the nearest geometry pixel
aaa, index = geom_tree.query(                       # I only need the index so I can define aaa as a dummy variable
    np.column_stack((lat_r, lon_r)),
    k=1
)
# To now assign the viewing zenith angles
vza_matched = vza_flat[index]
 
# Now it should be matched onto the same grid to make point by point comparisons possible
 
 
#%%
 
# For the sun-glint angles grid
SGA_flat = SGA.ravel()
SGA_matched = SGA_flat[index]
 
# I want to mask for only valid pixels
valid = (
    np.isfinite(R_flat) &
    np.isfinite(vza_matched) &
    np.isfinite(SGA_matched) &               	# Only for finite valid R, vza and SGA values
    (R_flat > 0) &
    (SGA_matched < 15)   # For the glint-dominated regime, only SGA<15 degrees
)
 
# Now I can redefine the valid variable pixels I will use
lat_use = lat_r[valid]
lon_use = lon_r[valid]
R_use   = R_flat[valid]
vza_use = vza_matched[valid]
 
 
 
#%%
 
# Now to make the wind estimates
wind_est = np.full(len(R_use), np.nan)   # To store the wind estimates
 
# For each valid pixel
for i in range(len(R_use)):
    theta = vza_use[i]  	# Set theta as the zenith angles
    R_obs = R_use[i]    	# For the observed reflectance
    # I will interpolate model reflectance at each zenith angle
    model_R = np.array([
        np.interp(theta, theta_common, ref_curves[w])
        for w in wind_speeds
    ])         	# I will interpolate the zenith angles for the same theta range
    # Now I can find the absolute difference between the Cox-Munk curve and the reflectance value
    diff = np.abs(model_R - R_obs)
    if np.all(np.isnan(diff)):   	# To check the differences are all valid
        continue
    # Finally I can estimate wind speeds by observing the nearest wind speed curve to the R value
    wind_est[i] = wind_speeds[np.nanargmin(diff)]
 
 
#%%
 
# Now I want to reconstruct a 2D spatial field so I can make a wind speed map
wind_map = np.full(R.shape, np.nan)	# I can make an array with the same shape as the reflectance data
wind_map_flat = wind_map.ravel()	# Now I can make the 2D map a 1D array so I can fill in the data
wind_map_flat[valid] = wind_est	# To fill in the valid wind speed estimates I have made
wind_map = wind_map_flat.reshape(R.shape)	# And now to convert back to the original grid shape for the wind map
  
# To store the wind estimate data:
wind_est1=wind_est
 
 
 
#%%
 
# Now I can plot the map for the 1D inversion as follows:
 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.feature import NaturalEarthFeature
plt.figure(figsize=(10, 8))
pro = ccrs.PlateCarree()
ax = plt.axes(projection=proj)
ax.set_extent([22.5, 27.5, 33.25, 36], crs=proj)  # For the limits
 
# For the colourmap
pcm = ax.pcolormesh(
    lon, lat, wind_map,
    transform=proj,
    cmap='inferno',        # A colour blind friendly colour map
    shading='auto'
)
cb = plt.colorbar(pcm, ax=ax, shrink=0.8)     # The colour bar
cb.set_label("Wind speed (m/s)", fontsize=16) # The colour bar label
 
# To add the real world features using Cartopy
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgrey', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.1)
 
# Now I can add the physcial mask
land_feature = NaturalEarthFeature(
    'physical',
    'land',
    '10m'
)
ax.add_feature(land_feature, facecolor='white', zorder=3)  # To cover the land with a white block masking
 
# To now add gridlines onto the plot
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
# To remove the top and right angles on the grid
gl.top_labels = False
gl.right_labels = False
 
# To manually add the axis labels to the plot
ax.text(0.5, -0.15, 'Longitude (°E)', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=18)
ax.text(-0.15, 0.5, 'Latitude (°N)', va='center', ha='right',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=18)
 
gl.xlabel_style = {'size': 14}  # Grid label fontsizes
gl.ylabel_style = {'size': 14}
 
plt.tight_layout()
plt.show()
 
 
 
 
#%%
# And now for the 2D version:
    
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
import scipy
# Path to home directory
HOME_DIR = '/…/' 
# 28 June reflectance (small regridded file)
REFL_FILE = HOME_DIR + 'MOD02QKM_2025-06-28_0815.nc'
 
# 28 June geometry
GEOM_FILE = HOME_DIR + 'GEOMETRY250_2025-06-28_0815.nc'
 
# To load the reflectance data
refl_ds = xr.open_dataset(REFL_FILE)
# Load geometry data
geom_ds = xr.open_dataset(GEOM_FILE)
    
lat = refl_ds['lat'].data
lon = refl_ds['lon'].data
 
# Again I will use band 2 for glint
R = refl_ds['band2_reflectance'].data
 
# To define the variables
sza = geom_ds['solar_zenith_angle'].data
saa = geom_ds['solar_azimuth_angle'].data
vza = geom_ds['satellite_zenith_angle'].data
vaa = geom_ds['satellite_azimuth_angle'].data
 
#%%
 
# Wind speeds corresponding to digitised values
wind_speeds = np.arange(0, 16)  # 0 to 15 m/s
 
# I will use the same interpolation of each Cox–Munk curve onto a common zenith grid
 
# Now as I need to consider the geometry, I need to factor in the peak of each curve
theta_peak = {}	# To create the empty list
for w in wind_speeds:	# For each of the wind speeds from 0 to 15 m/s
    y = ref_curves[w]   	# Define the y-values as the Cox-Munk curves reflectance
    index_peak = np.nanargmax(y)  	# To find where the peak reflectance is in the dataset
    theta_peak[w] = theta_common[index_peak]   # To find the corresponding peak theta value
theta_rel_curves = {}  # To create an empty list
 
# For each wind speed, I will find the relative angle distance a given pixel is from the peak sun-glint angle
for w in wind_speeds:  
    theta_rel_curves[w] = theta_common - theta_peak[w]  
    
    
    
    
    
    
#%%
# Now to ensure there is no resolution issues and everything is on the same grid
 
# Reflectance grid
lat_flat = refl_ds['lat'].values.ravel()
lon_flat = refl_ds['lon'].values.ravel()
R_flat = R.ravel()
vza_flat = vza.ravel()
SGA_flat = SGA.ravel()
 
# To set the latitude constraint for the Crete region I will observe
lat_min = 33.0
lat_max = 36.0
 
 
#%%
# For the mask
valid = (
    np.isfinite(R_flat) &
    np.isfinite(SGA_flat) &
    np.isfinite(vza_flat) & # For only valid pixels
 
    (R_flat > 0) & # With reflectance greater than 0
    (SGA_flat < 15) & # And SGA less than 15
    (lat_flat >= lat_min) &
    (lat_flat <= lat_max)
)
 
# Now to define the variables I will use after applying the masking
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
 
# Now I want to reconstruct a 2D spatial field so I can make a wind speed map
wind_map = np.full(R.shape, np.nan)	# I can make an array with the same shape as the reflectance data
wind_map_flat = wind_map.ravel()	# Now I can make the 2D map a 1D array so I can fill in the data
wind_map_flat[valid] = wind_est	# To fill in the valid wind speed estimates I have made
wind_map = wind_map_flat.reshape(R.shape)	# And now to convert back to the original grid shape for the wind map
 
# To store the wind estimate data for the 2D inversion:
wind_est2=wind_est
 
 
 
#%%
 
# And now again to plot the wind speed map:
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.feature import NaturalEarthFeature
 
plt.figure(figsize=(10, 8))
proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)
ax.set_extent([22.5, 27.5, 33.25, 36], crs=proj) # For the lat and lon limits of the map
 
# To make the colour map and bar
pcm = ax.pcolormesh(
    lon, lat, wind_map,
    transform=proj,
    cmap='inferno'
)
cb = plt.colorbar(pcm, ax=ax, shrink=0.8) # To add the colour bar
cb.set_label("Wind speed (m/s)", fontsize=16) # For the colour bar label
 
# To add the realworld features using Cartopy
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgrey', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.1)
 
# Now to add the physical land mask
land_feature = NaturalEarthFeature(
    'physical',
    'land',
    '10m'
)
ax.add_feature(land_feature, facecolor='white', zorder=3) # Mask a white block for land areas
 
# To add the grid boc
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}
# To remove the top and right angles on the grid
gl.top_labels = False
gl.right_labels = False
 
# And now finally to manually add the axis labels
ax.text(0.5, -0.15, 'Longitude (°E)', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=18)
ax.text(-0.15, 0.5, 'Latitude (°N)', va='center', ha='right',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=18)
 
plt.tight_layout()
plt.show()
 
 
#%%
# To print the stats for the mean wind speeds and St Dev for each model
print("Mean wind 1D model:", np.nanmean(wind_est1))
print("Std wind 1D model:", np.nanstd(wind_est1))
print("Mean wind 2D model:", np.nanmean(wind_est2))
print("Std wind 2D model:", np.nanstd(wind_est2)) 
 
