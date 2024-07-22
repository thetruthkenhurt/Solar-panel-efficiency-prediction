import sqlite3
import pandas as pd
import os 

def load_data(db_path):
    print(f"Loading data from: {db_path}")
    if not os.path.exists(f'{db_path}/weather.db') or not os.path.exists(f'{db_path}/air_quality.db'):
        print(f"Data files not found in {db_path}. Skipping data loading.")
        return None, None

    conn_weather = sqlite3.connect(f'{db_path}/weather.db')
    weather_df = pd.read_sql_query("SELECT * FROM weather", conn_weather)
    conn_weather.close()

    conn_air_quality = sqlite3.connect(f'{db_path}/air_quality.db')
    air_quality_df = pd.read_sql_query("SELECT * FROM air_quality", conn_air_quality)
    conn_air_quality.close()

    return weather_df, air_quality_df

def convert_date_columns(weather_df, air_quality_df):
    if weather_df is None or air_quality_df is None:
        print("Skipping date conversion due to missing data.")
        return None, None

    weather_df['date'] = pd.to_datetime(weather_df['date'], format='%d/%m/%Y')
    air_quality_df['date'] = pd.to_datetime(air_quality_df['date'], format='%d/%m/%Y')
    return weather_df, air_quality_df

if __name__ == "__main__":
    import os
    DB_PATH = os.getenv('DB_PATH', 'data')
    weather_df, air_quality_df = load_data(DB_PATH)
    weather_df, air_quality_df = convert_date_columns(weather_df, air_quality_df)
    print("Data Loaded and Date Columns Converted Successfully")
