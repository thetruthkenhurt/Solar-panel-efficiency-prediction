from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    print("ROC AUC Score:", auc)

if __name__ == "__main__":
    from train_model import train_model
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    evaluate_model(model, X_test, y_test)
    print("Model Evaluation Completed Successfully")
