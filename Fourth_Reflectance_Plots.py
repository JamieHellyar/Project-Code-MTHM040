import numpy as np
import xarray
import netCDF4
import scipy

# Path to home directory 
HOME_DIR = '/Users/Documents/Wind Project/’

# The easiest way to read a netCDF4 file in python is using the xarray library
file_path = HOME_DIR+'MOD02QKM_2025-06-26_0835.nc' 
dataset = xarray.open_dataset(file_path)

# To extract the numerical data for band 1
band1_data = dataset['band1_radiance'].data

# %%

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

# Now I will define the gaussian model so I can display in on my figure
def gaussian(x, a, mu, sigma):  
# Setting the inputs, x~Variable, a~Amplitude mu~Mean, sigma~StDev
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    # Formula for finding Gaussian function 
    # ~ note for myself np.exp is just finding exponential using NumPy

# Now I can try to add a exponential gaussian distribution
import scipy.special as sse

# To try and fit the exponential model:
def ExpNorm(x, a, x_0, mu, sigma, B):
    y1 = np.exp((mu/x_0) + (sigma**2/(2 * x_0**2)) - (x/x_0))
    y2 = sse.erfc(-(1/np.sqrt(2) * (((x - mu)/sigma) - (sigma/x_0))))
    return (a/(2 * x_0)) * y1 * y2 + B



# %%

# Now to make my reflectance plot

def exp_reflectance_plot(dataset, bands, target_lat, lon_min, lon_max):
    lat = dataset['lat'].data
    lon = dataset['lon'].data

    plt.figure(figsize=(10, 6))

   # To find nearest target latitude
    mean_lats = np.nanmean(lat, axis=1)
    row_index = np.argmin(np.abs(mean_lats - target_lat))

   # For each bandwidth
    for band in bands:
        data = np.ma.masked_equal(dataset[band].data, 0).astype(float)   # To mask out invalid data
       
        lat_r = data[row_index, :]    # To extract the latitude row
        lat_lon = lon[row_index, :]  # And then the corresponding longitudes

        mask = (lat_lon >= lon_min) & (lat_lon <= lon_max)	# To mask the longitude and latitude range
        x = lat_lon[mask]
        y = lat_r[mask]

        plt.plot(x, y, label="Raw Data", linewidth=1.5)		# To plot the raw data

        # Now, if the data exists I can estimate the parameters
        if np.any(y > 0):

            mu_est = np.sum(x * y) / np.sum(y)	# For the mean
            sigma_est = np.sqrt(np.sum(y * (x - mu_est)**2) / np.sum(y)) # For the standard deviation
            a_est = np.nanmax(y)	# For the max amplitude

            
            # Now to fit an EMG to the raw data
            try:
                p0 = [a_est, 1.0, mu_est, sigma_est, np.min(y)] # To make an array of the parameters
                params, _ = curve_fit(ExpNorm, x, y, p0=p0, maxfev=10000) # To run the EMG function
                y_emg = ExpNorm(x, params)	# For the plots y values
                plt.plot(x, y_emg, '-', label="EMG Distribution")	# To plot the EMG
            except RuntimeError:
                print(f"EMG fit failed for {band}")	# If the fit fails

    # For the labels
    plt.xlabel("Longitude (°E)",fontsize=18)
    plt.ylabel("Reflectance",fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()


# %%

# Now to call on the function and make the plot over the desired region far from any coastlines:    
exp_reflectance_plot(
    dataset,
    bands=['band1_reflectance'],
    target_lat=34.0,
    lon_min=17,
    lon_max=25
)











