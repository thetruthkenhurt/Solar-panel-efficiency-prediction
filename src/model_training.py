from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(X, y):
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Oversampling using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])
    print(report)
    return model

if __name__ == "__main__":
    from feature_engineering import feature_engineer, encode_target
    from data_loader import load_data, convert_date_columns
    from data_preprocessing import preprocess_data
    import os
    DB_PATH = os.getenv('DB_PATH', 'data')
    weather_df, air_quality_df = load_data(DB_PATH)
    weather_df, air_quality_df = convert_date_columns(weather_df, air_quality_df)
    weather_df_scaled, air_quality_df = preprocess_data(weather_df, air_quality_df)
    weather_df = feature_engineer(pd.DataFrame(weather_df_scaled, columns=weather_df.columns))
    weather_df = encode_target(weather_df)
    X = weather_df.drop(columns=['efficiency_label'])
    y = weather_df['efficiency_label']
    model = train_model(X, y)
    print("Model Training Completed Successfully")
