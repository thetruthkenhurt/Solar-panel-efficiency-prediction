import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def feature_engineer(weather_df, air_quality_df):
    # Combine weather and air quality data
    combined_df = pd.merge(weather_df, air_quality_df, on='date', how='inner')

    # Create a composite feature for total rainfall
    combined_df['total_rainfall'] = (combined_df['Daily Rainfall Total (mm)'] +
                                     combined_df['Highest 30 Min Rainfall (mm)'] +
                                     combined_df['Highest 60 Min Rainfall (mm)'] +
                                     combined_df['Highest 120 Min Rainfall (mm)'])

    # Create new feature: Temperature-Humidity Interaction
    combined_df['temp_humidity_interaction'] = combined_df['Maximum Temperature (deg C)'] * combined_df['Relative Humidity (%)']

    return combined_df

def encode_target(df):
    if 'Daily Solar Panel Efficiency' in df.columns:
        efficiency_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df['efficiency_label'] = df['Daily Solar Panel Efficiency'].map(efficiency_mapping)
    else:
        raise KeyError("The column 'Daily Solar Panel Efficiency' is missing from the DataFrame.")
    return df

def select_features(X, y, num_features):
    model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=num_features)
    fit = rfe.fit(X, y)
    selected_features = X.columns[fit.support_]
    return selected_features
