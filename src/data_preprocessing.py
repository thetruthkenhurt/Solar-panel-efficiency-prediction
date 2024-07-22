import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(weather_df, air_quality_df):
    # Handle missing values using forward fill
    weather_df.ffill(inplace=True)
    air_quality_df.ffill(inplace=True)

    # Replace special characters with NaN and convert to numeric types
    special_chars = ['--', '-', '']
    weather_df.replace(special_chars, pd.NA, inplace=True)
    air_quality_df.replace(special_chars, pd.NA, inplace=True)

    # Convert columns to numeric
    weather_numeric_cols = weather_df.columns.difference(['date', 'data_ref', 'Dew Point Category', 'Wind Direction', 'Daily Solar Panel Efficiency'])
    air_quality_numeric_cols = air_quality_df.columns.difference(['date', 'data_ref'])

    weather_df[weather_numeric_cols] = weather_df[weather_numeric_cols].apply(pd.to_numeric, errors='coerce')
    air_quality_df[air_quality_numeric_cols] = air_quality_df[air_quality_numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Fill remaining NaNs with column means for numeric columns
    weather_df[weather_numeric_cols] = weather_df[weather_numeric_cols].fillna(weather_df[weather_numeric_cols].mean())
    air_quality_df[air_quality_numeric_cols] = air_quality_df[air_quality_numeric_cols].fillna(air_quality_df[air_quality_numeric_cols].mean())

    # Ensure no infinite values
    weather_df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    air_quality_df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    weather_df.fillna(0, inplace=True)
    air_quality_df.fillna(0, inplace=True)

    # Standardize the numeric data
    scaler = StandardScaler()
    weather_scaled_df = pd.DataFrame(scaler.fit_transform(weather_df[weather_numeric_cols]), columns=weather_numeric_cols)
    air_quality_scaled_df = pd.DataFrame(scaler.fit_transform(air_quality_df[air_quality_numeric_cols]), columns=air_quality_numeric_cols)

    # Add non-numeric columns back to the scaled DataFrame
    weather_scaled_df['date'] = weather_df['date'].values
    weather_scaled_df['Daily Solar Panel Efficiency'] = weather_df['Daily Solar Panel Efficiency'].values
    air_quality_scaled_df['date'] = air_quality_df['date'].values

    return weather_scaled_df, air_quality_scaled_df
