In this GitHub repository you can find all the code I used for the results shown in the paper titled: "Using Sun-Glint Observations to Retrieve High-Resolution Near-Surface Ocean Wind Speed Estimates"

In this repository there are 8 files I used for different parts of the project.
The first file was used to interpolate the digitised Cox-Munk curves.
The second file contains all the code I used for my Froude number analysis.
The third file shows the code I used to calculate and plot the sun-glint angles around Crete.
The fourth file was used to plot reflectance curves with an EMG distribution.
The fifth file shows the inversion model I created to estimate wind speeds based on the observed reflectance and geometries.
The sixth file shows all the SARs related code I created to compare my estimates against.
The seventh file was the first file that looks at the UKESM simulations, with this file focusing on creating wind speed maps and measuring the variability of each pixel over time.
The eighth file contains the code I used to make statistical comparisons between my estimates and the UKESM simulations.

The datasets I used were as follows:
potential_temperature_1km_20250627T1800_12hr.nc - Which was simulated by the UKESM for this project
20250627T1800_1km_wind_speed_12hr.nc - Which was simulated by the UKESM for this project
20250627T1800_1km_land_binary_mask_12hr.nc - Which was simulated by the UKESM for this project
GEOMETRY250_2025-06-26_0835.nc - Which was retrieved from the MODIS observations, modified from https://satcorps.larc.nasa.gov/modis/
