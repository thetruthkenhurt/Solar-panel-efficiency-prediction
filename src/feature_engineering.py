import pandas as pd

def feature_engineer(weather_df):
    weather_df['temperature_range'] = weather_df['Maximum Temperature (deg C)'] - weather_df['Min Temperature (deg C)']
    weather_df['rainfall_intensity'] = weather_df['Daily Rainfall Total (mm)'] / weather_df['Sunshine Duration (hrs)']
    
    return weather_df

def encode_target(df):
    df['efficiency_label'] = pd.cut(df['Daily Solar Panel Efficiency'], bins=[-1, 0.33, 0.66, 1], labels=['Low', 'Medium', 'High'])
    df['efficiency_label'] = df['efficiency_label'].map({'Low': 0, 'Medium': 1, 'High': 2})
    return df

if __name__ == "__main__":
    from data_loader import load_data, convert_date_columns
    from data_preprocessing import preprocess_data
    import os
    DB_PATH = os.getenv('DB_PATH', 'data')
    weather_df, air_quality_df = load_data(DB_PATH)
    weather_df, air_quality_df = convert_date_columns(weather_df, air_quality_df)
    weather_df_scaled, air_quality_df = preprocess_data(weather_df, air_quality_df)
    weather_df = feature_engineer(pd.DataFrame(weather_df_scaled, columns=weather_df.columns))
    weather_df = encode_target(weather_df)
    print("Feature Engineering Completed Successfully")
