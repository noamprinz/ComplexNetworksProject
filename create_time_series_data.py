import numpy as np
import pandas as pd


def create_random_time_series(series_length: int = 1000, low=0, high=1, rolling_average_window:int = 0):
    """
    Creates a random time series with n data points
    :return: a random time series
    """
    if rolling_average_window == 0:
        values = np.random.uniform(low=low, high=high, size=series_length)
    else:
        values = np.random.uniform(low=low, high=high, size=series_length + rolling_average_window)
        values = pd.Series(values).rolling(rolling_average_window).mean()
        values = values[rolling_average_window:]
    return pd.Series(values)
