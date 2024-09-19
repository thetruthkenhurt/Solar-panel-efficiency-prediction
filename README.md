### Project Structure
```bash
├── data
│   ├── weather.db
│   └── air_quality.db
├── src
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── train_model.py
├── eda.ipynb
├── requirements.txt
├── run.sh
└── README.md
```
Please note that as the dataset is from a confidential source, it will not be provided in the repository. This is intended as a showcase of my work done implementing ML and DA techniques.

### Instructions to Run
1. Ensure you have all dependencies installed: `pip install -r requirements.txt`
2. Set the `DB_PATH` environment variable to the path of your data directory: `export DB_PATH=data` 
3. Run the pipeline: `bash run.sh`

## Pipeline Overview

1. **Data Loading (`data_loader.py`)**:
    - Load weather and air quality data from SQLite databases.

2. **Data Preprocessing (`data_preprocessing.py`)**:
    - Handle missing values, standardize numerical columns, and clean the data.

3. **Feature Engineering (`feature_engineering.py`)**:
    - Engineer new features and encode the target variable.

4. **Model Training & Evaluation (`train_model.py`)**:
    - Train machine learning models and evaluate their performance using various metrics.
	- Also includes code for hyperparameter tuning and training tuned models.
	
## Logical Steps/Flow of the Pipeline

1. **Data Loading**:
    - Connect to SQLite databases and load data into pandas DataFrames.

2. **Data Preprocessing**:
    - Clean and preprocess data, including handling missing values and standardizing numerical features.

3. **Feature Engineering**:
    - Create new features, such as `total_rainfall` and `temp_humidity_interaction`.
    - Encode the target variable `Daily Solar Panel Efficiency`.

4. **Model Training**:
    - Train models like Logistic Regression, Random Forest, XGBoost, and an Ensemble model.
    - Evaluate models using metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

5. **Model Evaluation**:
    - Compare model performances and select the best model based on evaluation metrics.

## Task 1: Exploratory Data Analysis (EDA)

### Preprocessing Steps
During the preprocessing step, we applied several transformations to clean and prepare the data for analysis and modeling. Key actions included:

1. **Handling Missing Values**:
   - Identified and addressed missing values in multiple columns using forward fill.

2. **Dealing with Special Characters and Negative Values**:
   - Processed & replaced special characters and replaced them with mean values for numeric columns and corrected negative values where they were not logically possible (e.g., wind speed).

3. **Outlier Detection and Treatment**:
   - Detected outliers in various columns and applied appropriate treatments, such as capping extreme outliers in `rainfall_intensity`.

4. **Standardizing Numeric Features**:
   - Standardized numeric features to ensure uniformity across the dataset.

5. **Extracting and Preserving Target Variable**:
   - Saved the 'Daily Solar Panel Efficiency' column before applying preprocessing transformations.

The cleaned dataset, free of missing values and outliers, is now ready for further analysis and modeling.

## Overview of Key Findings from EDA

Through this exploratory data analysis (EDA), we have gained several insights into the factors influencing solar panel efficiency:

1. **Data Structure and Quality:**
   - The weather and air quality datasets were successfully loaded and inspected.
   - Missing values were identified and handled appropriately.

2. **Key Variables:**
   - Several weather variables such as temperature, humidity, rainfall, and sunshine duration showed potential correlations with solar panel efficiency.
   - Air quality metrics like PM2.5 and PSI also indicated possible impacts on efficiency levels.

3. **Relationships and Patterns:**
   - Correlation analysis revealed significant relationships between certain weather conditions and solar panel efficiency.
   - Visualizations such as heatmaps and pair plots helped in understanding the interactions between different variables and efficiency.
   
4. **Imbalanced Dataset**
	- Dataset was imbalanced, with a higher proportion of 'Medium Efficiency' labels compared to 'Low Efficiency' and 'High Efficiency' labels.
	- This imbalance can lead to biased model predictions, where the model might favor the majority class.
	
5. **Feature Engineering:**
   - New features were created, such as total rainfall and temperature-humidity interaction, which could improve model predictions.

## Feature Engineering Summary

| Feature                         | Processing Steps                         |
|---------------------------------|------------------------------------------|
| `Daily Rainfall Total (mm)`     | Aggregated into `total_rainfall`         |
| `Maximum Temperature (deg C)`   | Used in `temp_humidity_interaction`      |
| `Relative Humidity (%)`         | Used in `temp_humidity_interaction`      |
| `Daily Solar Panel Efficiency`  | Encoded as target variable `efficiency_label` into '0','1','2'|

### Explanation for Selected Features
**Total Rainfall**: 
- We created a total_rainfall feature by aggregating different rainfall measurements, such as daily rainfall total, highest 30-minute rainfall, highest 60-minute rainfall, and highest 120-minute rainfall. This feature provides a comprehensive measure of the overall rainfall, which is expected to impact solar panel efficiency by affecting the amount of sunlight available.
**Temperature-Humidity Interaction**: 
- We introduced a temp_humidity_interaction feature by multiplying the maximum temperature with relative humidity. This feature captures the combined effect of temperature and humidity on solar panel efficiency. High temperatures with high humidity can reduce efficiency, and this interaction helps the model understand such complex relationships.

## Task 2: Machine Learning Pipeline 

### Addressing data imbalance 
To address this issue, we employed the following strategies:

**Stratified Splitting**:
- We used stratified train-test splits to ensure that the class distribution remains consistent across training and testing datasets. This helps in maintaining the representativeness of the data and ensures that the model is evaluated fairly across all classes.

**SMOTE (Synthetic Minority Over-sampling Technique)**: 
- We applied SMOTE to oversample the minority classes ('Low Efficiency' and 'High Efficiency'). SMOTE generates synthetic samples for the minority classes by interpolating between existing samples. This technique helps in balancing the dataset and provides the model with a more diverse training set, reducing the risk of overfitting to the majority class.

## Explanation of Choice of Models

1. **Logistic Regression**:
    - **Reason**: Provides a simple, interpretable baseline model. Helps in understanding the importance of features and serves as a benchmark for more complex models.

2. **Random Forest**:
    - **Reason**: Captures complex feature interactions and provides feature importance insights. Robust to overfitting and handles high-dimensional data well.

3. **XGBoost**:
    - **Reason**: Superior performance, especially with imbalanced datasets, due to its boosting and regularization capabilities. Known for improving predictive power through gradient boosting.

4. **Ensemble Model**:
    - **Reason**: Combines the strengths of multiple models (Logistic Regression, Random Forest, XGBoost) to improve overall performance. Offers better generalization by leveraging different model characteristics.

## Evaluation Metrics

For all models, the following evaluation metrics were used:

- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the ability to find all positive instances.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Assesses the model's ability to distinguish between classes.

### Summary of Evaluation Metrics

These evaluation metrics were chosen to provide a comprehensive understanding of the model's performance:

- **Accuracy** gives a general sense of performance but can be insufficient for imbalanced datasets.
- **Precision** and **Recall** are critical when the costs of false positives and false negatives differ, offering more nuanced insights than accuracy alone.
- **F1-Score** balances precision and recall, making it suitable for evaluating models on imbalanced datasets.
- **Confusion Matrix** provides a detailed view of the model's performance, highlighting the types of errors.
- **ROC AUC Score** offers an aggregate measure of performance across all classification thresholds, useful for model comparison.

## Model Evaluation Results

Here are the results of the best-performing models:

### XGBoost
- **Accuracy**: 0.90
- **Precision, Recall, and F1-Score**:
    - **Class 0 (Low Efficiency)**: Precision - 0.94, Recall - 0.82, F1-Score - 0.87
    - **Class 1 (Medium Efficiency)**: Precision - 0.90, Recall - 0.95, F1-Score - 0.93
    - **Class 2 (High Efficiency)**: Precision - 0.84, Recall - 0.84, F1-Score - 0.84
- **ROC AUC Score**: 0.9273591289552359


### Ensemble Model 
- **Accuracy**: 0.90
- **Precision, Recall, and F1-Score**:
    - **Class 0 (Low Efficiency)**: Precision - 0.94, Recall - 0.83, F1-Score - 0.88
    - **Class 1 (Medium Efficiency)**: Precision - 0.91, Recall - 0.95, F1-Score - 0.93
    - **Class 2 (High Efficiency)**: Precision - 0.84, Recall - 0.85, F1-Score - 0.84
- **ROC AUC Score**: 0.9209596510885284


#### Model Tuning 
We chose to tune the XGBoost model over the RandomForest model despite their similar untuned performance due to XGBoost's superior handling of imbalanced datasets and its ability to enhance predictive power through gradient boosting and regularization techniques.

Results of Tuning Hyperparameters for XGBoost model:

| Metric                             | Score                                      |
|------------------------------------|--------------------------------------------|
| Stratified K-Fold CV Scores        | 0.9093, 0.9111, 0.9004, 0.9119, 0.9155     |
| Mean CV Score                      | 0.9096                                     |

| Metric       | Score |
|--------------|-------|
| Precision    | 0.93  |
| Recall       | 0.79  |
| F1-Score     | 0.86  |
| Accuracy     | 0.88  |
| ROC AUC Score| 0.922 |

### Insights from Model Training & Tuning Phase:
 **Feature Engineering and Preprocessing Impact**:
   - Comprehensive feature engineering and preprocessing steps have significantly contributed to model performance and predictive accuracy. These steps often have a more substantial impact than hyperparameter tuning, as they enhance the quality and relevance of the input data.

## Other Considerations for Deployment

1. **Computational Resources**:
   - Consider the computational cost and resources required for training and deploying models. For instance, XGBoost and the Ensemble model may require more computational power and time compared to Logistic Regression.

2. **Training Time**:
   - Models like XGBoost and Ensemble typically take longer to train due to their complexity. Ensure adequate computational infrastructure to handle the training process efficiently.

3. **Real-time Predictions**:
   - Evaluate the inference time of each model to ensure they meet real-time prediction requirements. Logistic Regression as a basic model will generally perform faster, while XGBoost and Ensemble models might need optimization for real-time applications.


