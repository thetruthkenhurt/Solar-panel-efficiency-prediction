# Inside data_preprocessing.py

def preprocess_data(weather_df, air_quality_df):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Handle missing values using forward fill
    weather_df.ffill(inplace=True)
    air_quality_df.ffill(inplace=True)

    # Replace special characters with NaN and convert to numeric types
    special_chars = ['--', '-', '']

    weather_numeric_cols = [
        'Daily Rainfall Total (mm)',
        'Highest 30 Min Rainfall (mm)',
        'Highest 60 Min Rainfall (mm)',
        'Highest 120 Min Rainfall (mm)',
        'Min Temperature (deg C)',
        'Maximum Temperature (deg C)',
        'Min Wind Speed (km/h)',
        'Max Wind Speed (km/h)',
        'Sunshine Duration (hrs)',
        'Cloud Cover (%)',
        'Relative Humidity (%)',
        'Wet Bulb Temperature (deg F)',
        'Air Pressure (hPa)'
    ]

    for col in weather_numeric_cols:
        weather_df[col] = weather_df[col].replace(special_chars, pd.NA)
        weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')

    air_quality_numeric_cols = [
        'pm25_north',
        'pm25_south',
        'pm25_east',
        'pm25_west',
        'pm25_central',
        'psi_north',
        'psi_south',
        'psi_east',
        'psi_west',
        'psi_central'
    ]

    for col in air_quality_numeric_cols:
        air_quality_df[col] = air_quality_df[col].replace(special_chars, pd.NA)
        air_quality_df[col] = pd.to_numeric(air_quality_df[col], errors='coerce')

    # Reapply forward fill after conversion to handle any new NaNs
    weather_df.ffill(inplace=True)
    air_quality_df.ffill(inplace=True)

    # Ensure no infinite values
    weather_df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    air_quality_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

    # Fill remaining NaNs with column means
    numeric_weather_df = weather_df.select_dtypes(include=[np.number])
    weather_df[numeric_weather_df.columns] = numeric_weather_df.fillna(numeric_weather_df.mean())
    
    numeric_air_quality_df = air_quality_df.select_dtypes(include=[np.number])
    air_quality_df[numeric_air_quality_df.columns] = numeric_air_quality_df.fillna(numeric_air_quality_df.mean())

    # Ensure no infinite values again after filling NaNs
    weather_df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    air_quality_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

    # Replace remaining NaNs with zeros (or any other value you find appropriate)
    weather_df.fillna(0, inplace=True)
    air_quality_df.fillna(0, inplace=True)

    # Feature engineering
    weather_df['temperature_range'] = weather_df['Maximum Temperature (deg C)'] - weather_df['Min Temperature (deg C)']
    weather_df['rainfall_intensity'] = weather_df['Daily Rainfall Total (mm)'] / weather_df['Sunshine Duration (hrs)']

    # Replace infinite values in rainfall_intensity with a large number
    weather_df['rainfall_intensity'].replace([np.inf, -np.inf], 1e6, inplace=True)

    # Debugging: Check for negative values and replace them with appropriate values
    weather_df['Min Wind Speed (km/h)'] = weather_df['Min Wind Speed (km/h)'].apply(lambda x: max(x, 0))
    weather_df['Max Wind Speed (km/h)'] = weather_df['Max Wind Speed (km/h)'].apply(lambda x: max(x, 0))

    # Select only numeric columns for scaling
    weather_df_numeric = weather_df.select_dtypes(include=[np.number])

    # Standardize the numeric data
    scaler = StandardScaler()
    weather_df_scaled = scaler.fit_transform(weather_df_numeric)

    # Convert back to DataFrame with original columns
    weather_df_scaled = pd.DataFrame(weather_df_scaled, columns=weather_df_numeric.columns)

    return weather_df_scaled, air_quality_df
