import sqlite3
import pandas as pd

def load_data(db_path):
    conn_weather = sqlite3.connect(f'{db_path}/weather.db')
    weather_df = pd.read_sql_query("SELECT * FROM weather", conn_weather)
    conn_weather.close()

    conn_air_quality = sqlite3.connect(f'{db_path}/air_quality.db')
    air_quality_df = pd.read_sql_query("SELECT * FROM air_quality", conn_air_quality)
    conn_air_quality.close()

    return weather_df, air_quality_df

def convert_date_columns(weather_df, air_quality_df):
    weather_df['date'] = pd.to_datetime(weather_df['date'], format='%d/%m/%Y')
    air_quality_df['date'] = pd.to_datetime(air_quality_df['date'], format='%d/%m/%Y')
    return weather_df, air_quality_df

if __name__ == "__main__":
    import os
    DB_PATH = os.getenv('DB_PATH', 'data')
    weather_df, air_quality_df = load_data(DB_PATH)
    weather_df, air_quality_df = convert_date_columns(weather_df, air_quality_df)
    print("Data Loaded and Date Columns Converted Successfully")
