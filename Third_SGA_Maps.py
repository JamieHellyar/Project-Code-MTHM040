import numpy as np
import xarray
import netCDF4
import scipy

# Path to home directory
HOME_DIR = '/Users/Jamie/Documents/Wind Project/'

# I can read the netCDF4 file in python is using the xarray library
file_path = HOME_DIR+'GEOMETRY250_2025-06-26_0835.nc'
dataset = xarray.open_dataset(file_path)

# %%

# To calculate the sun glint angles I can define the following function

def GlintAngle(solarzenith,
               solarazimuith,
               satzenith,
               satazimuth):
    
    
    d = np.deg2rad(abs(satazimuth - solarazimuith))
    
    a = np.cos(np.deg2rad(satzenith)) * np.cos(np.deg2rad(solarzenith))
    
    b = np.sin(np.deg2rad(satzenith)) * np.sin(np.deg2rad(solarzenith)) * np.cos(d)
    
    glint_radiance=np.arccos(a - b)
    
    return np.rad2deg(glint_radiance)

# Now to compute glint angles from my dataset variables
glint_angle = GlintAngle(dataset['solar_zenith_angle'].data,
                                  dataset['solar_azimuth_angle'].data,
                                  dataset['satellite_zenith_angle'].data,
                                  dataset['satellite_azimuth_angle'].data)


dataset['glint_angle'] = (('y', 'x'), glint_angle)




# %%

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# First I will define the longitude and latitude data
lon = dataset['lon'].data
lat = dataset['lat'].data

# Then I can create the coordinate map 
plt.figure(figsize=(10, 8))
proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)

# Now to plot the data
pcm = ax.pcolormesh(lon, lat, glint_angle, 
                    transform=proj, cmap='viridis', shading='auto')
# This creates a colour map for each longitude and latitude point



# For the colour bar
cb = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.8) 
cb.set_label('Glint angle (°)', fontsize=16) # For the colour bar label

# Using Cartopy I can add these features to my map
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black')
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgrey', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.1)



# Now to add contours for the areas with the lowest angles at 1, 5, 10, 20 and 30 degrees
contours = ax.contour(lon, lat, glint_angle, 
                      transform=proj, colors='white', linewidths=0.1,
                      levels=[1, 5, 10, 20, 30])


# To add angles to the contours on the map manually
ax.clabel(
    contours,
    inline=True,
    fmt=lambda x: f"{x:.2f}°",
    fontsize=14,
    colors='white',
    manual=[(20,35), (22,39), (22,36), (24,39), (25,36)]
)

# To add the gridlines
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

# To remove the top and right grid labels
gl.top_labels = False
gl.right_labels = False

# To manually add the axis labels
ax.text(0.5, -0.15, 'Longitude (°E)', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=18)
ax.text(-0.1, 0.65, 'Latitude (°N)', va='center', ha='right',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=18)

gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}


plt.tight_layout()
plt.show()












