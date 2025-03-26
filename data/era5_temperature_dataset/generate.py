import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import random
import math


# =============================================================================
# 1. Download Data using the Copernicus Climate Data Store API
# =============================================================================
def download_era5_t2m(output_file='era5_t2m.nc'):
    request = {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": [
            "1979", "1980", "1981",
            "1982", "1983", "1984",
            "1985", "1986", "1987",
            "1988", "1989", "1990",
            "1991", "1992", "1993",
            "1994", "1995", "1996",
            "1997", "1998", "1999",
            "2000", "2001", "2002",
            "2003", "2004", "2005",
            "2006", "2007", "2008",
            "2009", "2010", "2011",
            "2012", "2013", "2014",
            "2015", "2016", "2017",
            "2018", "2019", "2020",
            "2021", "2022", "2023",
            "2024", "2025"
        ],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": ["00:00", "08:00", "16:00"],
        "data_format": "netcdf",
    }


    c = cdsapi.Client()
    # Here we request hourly data for a single day; you may loop over multiple days
    c.retrieve(
        'reanalysis-era5-single-levels',
        request,
        output_file)
    print(f"Downloaded ERA5 data (8-hourly)")


# =============================================================================
# 2. Helper Functions for Preprocessing and Encoding
# =============================================================================
def encode_time(dt):
    """
    Given a datetime object, compute the sine and cosine encodings for
    hour-of-day and day-of-year.
    """
    hour = dt.hour
    day_of_year = dt.timetuple().tm_yday
    sin_hour = math.sin(2 * math.pi * hour / 24)
    cos_hour = math.cos(2 * math.pi * hour / 24)
    sin_day = math.sin(2 * math.pi * day_of_year / 365)
    cos_day = math.cos(2 * math.pi * day_of_year / 365)
    return [sin_hour, cos_hour, sin_day, cos_day]

def encode_coordinates(lat, lon):
    """
    Encode latitude and longitude into [sin(latitude), sin(longitude), cos(longitude)].
    The input lat/lon are assumed to be in degrees.
    """
    # Convert degrees to radians for the trigonometric functions
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    return [math.sin(lat_rad), math.sin(lon_rad), math.cos(lon_rad)]


# =============================================================================
# 3. Construct a Single Sample from the Dataset
# =============================================================================
def create_sample(ds):
    """
    Creates one sample from the ERA5 dataset.
    
    Parameters:
        ds (xarray.Dataset): The dataset loaded from the ERA5 NetCDF file.
        
    Returns:
        sample (dict): A dictionary with keys "description" and "data" where:
            - "description" contains:
                - "coords": a list of encoded coordinates for each station
                - "start": the encoded start time [sin(hour), cos(hour), sin(day), cos(day)]
            - "data": a list of lists, each inner list is the normalized temperature 
                      time series for one station.
    """
    # Ensure the time dimension is sorted and available
    times = ds.valid_time.values
    total_time_points = len(times)
    
    # Choose a random window length between 8 and 16 (corresponding roughly to 2–4 days)
    window_length = random.randint(8, 16)
    
    # Choose a random starting index such that the window is within the data bounds
    start_idx = random.randint(0, total_time_points - window_length)
    
    # Get the start time as a datetime object (assuming times are in a datetime64 format)
    start_time = np.datetime64(times[start_idx]).astype('M8[s]').astype(datetime)
    
    # Encode start time features
    start_encoding = encode_time(start_time)
    
    # Total available spatial grid (assume ds has 1D lat and lon arrays)
    lats = ds.latitude.values
    lons = ds.longitude.values
    num_lat = len(lats)
    num_lon = len(lons)
    total_grid_points = num_lat * num_lon
    
    # Determine number of reporting stations for this sample (randomly between 60 and 90)
    num_stations = random.randint(60, 90)
    
    # Sample unique grid indices from the flattened grid
    sampled_indices = random.sample(range(total_grid_points), num_stations)
    
    # Prepare lists to hold station coordinate encodings and temperature time series data
    coords_list = []
    data_list = []
    
    # Get the t2m data for the chosen time window
    # Convert from Kelvin to Celsius by subtracting 273.15
    t2m_window = ds['t2m'][start_idx:start_idx+window_length, :, :].values - 273.15
    
    # For each sampled grid point, extract the station time series and spatial features
    for idx in sampled_indices:
        # Convert flat index to 2D grid indices (i, j)
        i, j = np.unravel_index(idx, (num_lat, num_lon))
        lat_val = lats[i]
        lon_val = lons[j]
        # Spatial encoding for this station
        coord_enc = encode_coordinates(lat_val, lon_val)
        coords_list.append(coord_enc)
        
        # Extract the temperature time series for this station (for the selected window)
        station_series = t2m_window[:, i, j].tolist()
        data_list.append(station_series)
    
    # Concatenate all station series to compute the sample-wise mean and std for normalization
    flat_values = np.concatenate(data_list)
    mean_val = np.mean(flat_values)
    std_val = np.std(flat_values)
    
    # Avoid division by zero
    if std_val == 0:
        std_val = 1.0
    
    # Normalize each station's time series
    normalized_data = []
    for series in data_list:
        norm_series = [ (val - mean_val) / std_val for val in series ]
        normalized_data.append(norm_series)
    
    # Assemble the sample JSON structure
    sample = {
        "description": {
            "coords": coords_list,  # List of [sin(lat), sin(lon), cos(lon)] for each station
            "start": start_encoding  # [sin(hour), cos(hour), sin(day_of_year), cos(day_of_year)]
        },
        "data": normalized_data  # List of time series (each is a list of normalized temperature values)
    }
    return sample

# =============================================================================
# 4. Assembling the Complete Dataset and Splitting into Train/Val/Test
# =============================================================================
def assemble_dataset(ds, num_samples_train, num_samples_val, num_samples_test):
    """
    Generate a set of samples from the dataset and split into training, 
    validation, and test sets.
    
    For demonstration, the default num_samples is set low. In the xVal paper,
    they generated 1.25 million samples with splits:
      - Training: 1,000,000 samples
      - Validation: 125,000 samples
      - Test: 125,000 samples
      
    Parameters:
        ds (xarray.Dataset): The loaded ERA5 dataset.
        num_samples (int): Total number of samples to generate.
        
    Returns:
        splits (dict): Dictionary with keys 'train', 'val', 'test' each containing a list of samples.
    """
    num_samples = num_samples_train + num_samples_val + num_samples_test

    samples = []
    for _ in range(num_samples):
        sample = create_sample(ds)
        samples.append(sample)

    dataset_splits = {
        "train": samples[:num_samples_train],
        "val": samples[num_samples_train:num_samples_train+num_samples_val],
        "test": samples[num_samples_train+num_samples_val:]
    }
    return dataset_splits


if __name__ == "__main__":
    netcdf_file = 'era5_t2m.nc'
    download_era5_t2m(netcdf_file)

    ds = xr.open_dataset(netcdf_file)

    dataset_splits = assemble_dataset(ds, num_samples_train=800, num_samples_val=100, num_samples_test=100)

    # Save the splits to JSON files (one file per split)
    for split_name, samples in dataset_splits.items():
        with open(f"{split_name}_samples.json", "w") as outfile:
            json.dump(samples, outfile)
        print(f"Saved {len(samples)} samples to {split_name}_samples.json")