import xarray as xr
import matplotlib.pyplot as plt

# First to load the datafile for the model temperature data

HOME_DIR = '/Users/Documents/Wind Project/Model Data/'

# To define the potential temperatures

theta_ds = xr.open_dataset(HOME_DIR + 'potential_temperature_1km_20250627T1800_12hr.nc')

# Now to load the datafile for the model wind speeds

HOME_DIR2 = '/Users/Documents/Wind Project/Model Data/1km/'

# To define the wind speeds

wind_ds = xr.open_dataset(HOME_DIR2 + '20250627T1800_1km_wind_speed_12hr.nc')

# And now to define the altitudes

land = xr.open_dataset(HOME_DIR2 + '20250627T1800_1km_land_binary_mask_12hr.nc')




#%%

# Ok so now I want to calculate the Froude number for the island of Crete

import numpy as np

# So first I need to find the temperature data times nearest to 09:00
theta_09 = theta_ds.sel(time='2025-06-28T09:00:00', method='nearest')

# And now I need to find the wind speed data times nearest to 09:00 to check they match
wind_09 = wind_ds.sel(time='2025-06-28T09:00:00', method='nearest')

# To check these values
print(theta_09.time.values)
print(wind_09.time.values)

#%%

# Now I want to define the variables, using data from the time nearest to 09:00

theta = theta_09['potential_temperature']  # Pressure, latitude and longitude
pressure = theta_09['pressure'] * 100      # To convert from hPa to Pa
U = wind_09['wind_speed']                  # In latitude and longitudes
h = theta_09['surface_altitude']           # In latitude and longitudes
land_mask = land['land_binary_mask']       # To define the land mask
U = U.interp_like(h)                       # To ensure U and h are on the same grid




#%%

# Currently my potential temperatures are in terms of pressure levels,
# so I need to find the geometric heights for each pressure level to make 
# accurate comparisons with my altitude data

# To define the constant I will use
g = 9.81
Rd = 287.0

# To define the air temperature
T = theta_09['air_temperature']

# Surface pressure
ps = theta_09['surface_air_pressure']


# To calculate height of pressure levels
z = (Rd * T / g) * np.log(ps / pressure)


#%%

# Ok and to find the Brunt-Vaisala frequency I need to know dtheta/dz

# So first I will sort the pressure data from lowest to highest
theta_sorted = theta.sortby('pressure', ascending=False)
z_sorted = z.sortby('pressure', ascending=False)



#And now to compute the derivative
dtheta = theta_sorted.diff('pressure')
dz = z_sorted.diff('pressure')

dtheta_dz = dtheta / dz

theta_new = theta_sorted.isel(pressure=slice(1, None))
N2 = (g / theta_new) * dtheta_dz

#%%

# To calculate the Brunt-Vaisala:

N2 = N2.where(N2 > 0)

N = np.sqrt(N2)

# For the lowest values nearer the surface:
    
pressure_new = theta_sorted.pressure.isel(pressure=slice(1, None))

# To reorder the N values according to the new pressure array
N = N.assign_coords(pressure=pressure_new)

# Select lower layer
N_low = N.sel(pressure=slice(950, 850)).mean(dim='pressure')




#%%

# To now calculate the Froude numbers:

Fr = U / (N_low * h)


#%%

# Now to mask low values: 
    
Fr = Fr.where(h > 50)           # To remove near-zero terrain
Fr = Fr.where(np.isfinite(Fr))  # To remove infinte values

land_mask = land['land_binary_mask']	# To apply the land mask

Fr_land = Fr.where(land_mask == 1)


#%%

# Now I can plot the Froude number map showing if Fr values are >1, <1 or =1

# To define the tolerance
eps = 0.05

# And now to assign the regimes

Fr_regime = xr.where(Fr_land < 1 - eps, 0,
              xr.where(abs(Fr_land - 1) <= eps, 1,
              xr.where(Fr_land > 1 + eps, 2, np.nan)))




#%%

# Ok and now to shrink the data to around Crete:
    
Fr_crete = Fr_land.sel(
    longitude=slice(23.25, 27),
    latitude=slice(34.75, 35.75)
)

h_crete = h.sel(
    longitude=slice(23.25, 27),
    latitude=slice(34.75, 35.75)
)

Fr_regime_crete = Fr_regime.sel(
    longitude=slice(23.25, 27),
    latitude=slice(34.75, 35.75)
)


#%%

from matplotlib.colors import ListedColormap

plt.figure(figsize=(14,6))

cmap = ListedColormap(['blue', 'yellow', 'red'])		# For the colour map

im = plt.pcolormesh(
    Fr_regime_crete.longitude,
    Fr_regime_crete.latitude,
    Fr_regime_crete,
    cmap=cmap,
    shading='auto'
)

# To add land contours on the plot:
plt.contour(
    h_crete.longitude,
    h_crete.latitude,
    h_crete,
    levels=[250, 500, 1000, 1500, 2000],		# Chosen altitudes for the plot
    colors='black',
    linewidths=0.7
)

# For the colour bar and labels
cbar = plt.colorbar(im, ticks=[0,1,2])
cbar.ax.set_yticklabels(['Fr<1, Blocked', 'Fr≈1, Critical', 'Fr>1, Flow over'], fontsize=16)
plt.xlabel('Longitude', fontsize=18)
plt.ylabel('Latitude', fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()


#%%

# For some statistics about the regimes

blocked = (Fr_regime_crete == 0).sum() / np.isfinite(Fr_regime_crete).sum()
critical = (Fr_regime_crete == 1).sum() / np.isfinite(Fr_regime_crete).sum()
flow = (Fr_regime_crete == 2).sum() / np.isfinite(Fr_regime_crete).sum()

print("Blocked fraction:", float(blocked.values))
print("Critical fraction:", float(critical.values))
print("Flow over fraction:", float(flow.values))










