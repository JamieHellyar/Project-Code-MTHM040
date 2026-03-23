import iris
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import numpy as np

# To extract the model data files
ROOT = '/Users/Jamie/Documents/Wind Project/Model Data/'
filepath_333m = ROOT+'333m/{}_333m_wind_speed_12hr.nc'
filepath_1km = ROOT+'1km/{}_1km_wind_speed_12hr.nc'
filepath_global = ROOT+'global/{}_global_wind_speed_12hr.nc'

# For the test day of the 28th of June (I will repeat the same exact code for the 23rd and the 3rd)
date = '20250627T1800'

# I can load the data for each resolution using iris
data_333m = iris.load_cube(filepath_333m.format(date))
data_1km = iris.load_cube(filepath_1km.format(date))
data_glob = iris.load_cube(filepath_global.format(date))

# And now to convert the global data to -180 to 180 to be consistent with everything else
lon = data_glob.coord('longitude')
lon_wrap = ((lon.points+180)%360)-180 # To convert the coordinates

# And now I can reorder the data as Iris requires monotonic coordinates
sorted_idx = np.argsort(lon_wrap) 
lon.points = lon_wrap[sorted_idx] 

# Now I can find axis that corresponds to the new longitude data
lon_axis = data_glob.coord_dims(lon)[0] 
data_glob.data=np.take(data_glob.data,sorted_idx,axis=lon_axis) # To reorder the data to match the reordered longitude coordinate
lon.circular=True

# Now I want to get the dates and convert from standard time to datetime
time_coord = data_1km.coord('time')
times = time_coord.units.num2date(time_coord.points)

# I also need to confirm that the dates match across the different resolution files
time_coord_333m = data_333m.coord('time')
assert np.all(times == time_coord_333m.units.num2date(time_coord_333m.points))

# And now I can save the coordinates for each resolution
lats_333m = data_333m.coord('latitude').points
lons_333m = data_333m.coord('longitude').points

# To define for 1km as well
lats_1km = data_1km.coord('latitude').points
lons_1km = data_1km.coord('longitude').points

# And now for the 10km resolutions
lats_glob = data_glob.coord('latitude').points
lons_glob = data_glob.coord('longitude').points

# Finally I can save data of the  time slice required to numpy array for each resolution
i=0
ws_333m = np.array(data_333m.data[i,:,:]) # The dimensions are time, lat, lon so I need to select the time slice but keep the full spatial domain
ws_1km = np.array(data_1km.data[i,:,:])	   # For each resolution
ws_glob = np.array(data_glob.data[i,:,:])




#%%

#First want to look for a time around 08:15 as that was the time of the MODIS overpass

from datetime import datetime
import numpy as np

# To define the target time
target_time = datetime(2025, 6, 28, 8, 15)

# Now I can look at the difference in each time from the target times
time_diffs = np.array([abs((t - target_time).total_seconds()) for t in times])


# And now to find the smallest time difference
i = np.argmin(time_diffs)

# And now I can print the closest model time
print("Closest model time:", times[i])

# This gives 08:11
# Now I have index i that I can use to make the plots


#%%

# To now define the wind speed arrays for each resoluiton
ws_333m = np.array(data_333m.data[i, :, :])
ws_1km  = np.array(data_1km.data[i, :, :])
ws_glob = np.array(data_glob.data[i, :, :])

# For the max and min lat and lon ranges
lat_min, lat_max = 33, 36
lon_min, lon_max = 23, 28

# Now I can define a subset for these to apply the mask for the longitudes and latitude ranges
def subset(ws, lats, lons):
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    return ws[np.ix_(lat_mask, lon_mask)], lats[lat_mask], lons[lon_mask]

# And to apply this to each resolution from the model
ws333, lat333, lon333 = subset(ws_333m, lats_333m, lons_333m)	# 333 m
ws1k,  lat1k,  lon1k  = subset(ws_1km,  lats_1km,  lons_1km)	# 1 km
wsg,   latg,   long   = subset(ws_glob, lats_glob, lons_glob)	# 10 km/global



#%%

# Now to plot:
proj = ccrs.PlateCarree()
fig, axes = plt.subplots(
    1, 3, figsize=(15, 5),		# To plot 3 figures
    subplot_kw={'projection': proj},
    constrained_layout=True
)

# I need to define which data I will plot, for all 3 resolutions
data   = [(ws333, lat333, lon333),
          (ws1k,  lat1k,  lon1k),
          (wsg,   latg,   long)]


# To define the title for each subplot to show resolutions
titles = ['333 m model', '1 km model', '10 km model']

# For each subplot
for ax, (ws, lat, lon), title in zip(axes, data, titles):		# Zipping each resolutions subplot parts together
    pcm = ax.pcolormesh(
        lon, lat, ws,			# For the grid coordinates and the wind speeds to be plotted
        cmap='inferno',			# FOr the colour scheme
        vmin=0, vmax=15,			# And the colour scale
        shading='auto',
        transform=ccrs.PlateCarree()
    )
    ax.set_title(title,fontsize=20)		# For the subplot titles
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])	# To ensure all plots are focused on the same region
    ax.coastlines(resolution='10m')		# To add the coastlines
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=0)

# For the colour bar
cbar = fig.colorbar(pcm, ax=axes, orientation='vertical', shrink=0.85)
cbar.set_label('Wind speed (m/s)', fontsize=16)


plt.show()








#%%

# I now want to look at the variability over different times to see which 
# regions change most frequently

# For the 1km model:

# Convert entire cube to numpy
ws_all = np.array(data_1km.data)  # shape: (time, lat, lon)

# Now I will compute the temporal standard deviation at each grid point
ws_std = np.std(ws_all, axis=0)


#%%

# And now the same again for the 333m model

# Convert full 333 m cube to numpy
ws_all_333 = np.array(data_333m.data)   # Which has the shape (time, lat, lon)

# For the temporal statistics
ws_std_333  = np.std(ws_all_333, axis=0)
ws_mean_333 = np.mean(ws_all_333, axis=0)




#%%

# To focus on just Crete:
    
# To define the lat lon range
lat_min, lat_max = 34, 36
lon_min, lon_max = 23, 27

# Now to create the masks
lat_mask = (lats_333m >= lat_min) & (lats_333m <= lat_max)
lon_mask = (lons_333m >= lon_min) & (lons_333m <= lon_max)

# Now I can apply the masks to the 333 m resolution st dev and latitude and longitudes to create new subsets
std_sub = ws_std_333[np.ix_(lat_mask, lon_mask)]
lat_sub = lats_333m[lat_mask]
lon_sub = lons_333m[lon_mask]
ws_sub = ws_333m[np.ix_(lat_mask, lon_mask)] 

# The same again but for all wind speeds:
ws_all_333 = np.array(data_333m.data)
ws_all_sub = ws_all_333[:, lat_mask, :][:, :, lon_mask]

# And for the wind speed standard deviation for the subset
ws_std_sub = np.std(ws_all_sub, axis=0)



#%%

# And to mask out the highest variability regions:     
threshold = np.percentile(ws_std_sub, 90)	# To define the threshold
high_var_mask = ws_std_sub > threshold	# Hence to apply the threshold

# I can now filter the wind speeds with this threshold masking
ws_filtered = np.where(high_var_mask, np.nan, ws_sub)

print("90th percentile threshold:", threshold)

# And to apply the mask to wind speed field
ws_filtered = np.where(high_var_mask, np.nan, ws_sub)












