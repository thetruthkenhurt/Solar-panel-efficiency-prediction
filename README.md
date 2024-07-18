# AIAP 18 Technical Assessment

## Full Name: Goh Joon Sian Kenneth

## Email: kennethgjs@gmail.com

### Project Structure
- `src/`: Contains all Python scripts for the pipeline
- `run.sh`: Bash script to execute the pipeline
- `requirements.txt`: List of dependencies
- `eda.ipynb`: Jupyter notebook for exploratory data analysis

### Instructions to Run
1. Ensure you have all dependencies installed: `pip install -r requirements.txt`
2. Set the `DB_PATH` environment variable to the path of your data directory: `export DB_PATH=data`
3. Run the pipeline: `bash run.sh`

### Pipeline Overview
- **Data Loading**: Scripts to load data from SQLite databases.
- **Data Preprocessing**: Scripts to clean and preprocess the data.
- **Feature Engineering**: Scripts to engineer features and encode target.
- **Model Training**: Scripts to train machine learning models.
- **Model Evaluation**: Scripts to evaluate the performance of models.

### Preprocessing Steps
During the preprocessing step, we applied several transformations to clean and prepare the data for analysis and modeling. Key actions included:

1. **Handling Missing Values**:
   - Identified and addressed missing values in multiple columns.

2. **Dealing with Infinite and Negative Values**:
   - Replaced infinite values and corrected negative values where they were not logically possible (e.g., wind speed).

3. **Outlier Detection and Treatment**:
   - Detected outliers in various columns and applied appropriate treatments, such as capping extreme outliers in `rainfall_intensity`.

4. **Standardizing Numeric Features**:
   - Standardized numeric features to ensure uniformity across the dataset.

5. **Extracting and Preserving Target Variable**:
   - Saved the 'Daily Solar Panel Efficiency' column before applying preprocessing transformations.

The cleaned dataset, free of missing values and outliers, is now ready for further analysis and modeling.

### Key Findings from EDA
In the exploratory data analysis (EDA), we have identified key weather features that significantly impact solar panel efficiency. Here are some key insights:

- **Total Rainfall**: Higher total rainfall is associated with lower solar panel efficiency. This relationship suggests that rainy days with less sunlight negatively impact solar panel performance.
- **Sunshine Duration**: Higher sunshine duration is positively correlated with higher solar panel efficiency, underscoring the importance of sunlight availability for solar power generation.
- **Max Wind Speed**: While max wind speed did not show a strong direct correlation with total rainfall, understanding wind conditions can provide additional context to weather patterns affecting solar efficiency.
- **Efficiency Labels**: The efficiency labels were imbalanced, with the majority of data points falling into the medium efficiency category. This class imbalance needs to be addressed during model training to ensure balanced performance.

### Feature Engineering
Based on the insights from the EDA, we performed feature engineering to create meaningful features:
- **Total Rainfall**: Combined various rainfall measurements into a composite feature.
- **Sunshine Duration**: Utilized as a key feature due to its positive correlation with efficiency.
- **Max Wind Speed**: Included to provide context to weather conditions.
- **Temperature Range**: Calculated as the difference between maximum and minimum temperature.
- **Rainfall Intensity**: Created as a feature to capture the intensity of rainfall.

### Model Evaluation
As we move on to the machine learning pipeline, we will focus on:
1. **Addressing Class Imbalance**: Implement techniques such as resampling or class weighting to handle the imbalance in efficiency labels.
2. **Evaluating Feature Engineering**: Review the features we have created and consider potential enhancements to capture more complex relationships between weather conditions and solar efficiency, if necessary.
3. **Model Training and Evaluation**: Build and evaluate machine learning models to predict solar panel efficiency, using robust methods to handle outliers and non-linear relationships.

### Stratification and Sampling
- **Stratified Split**: Used to maintain class distribution in training and testing sets.
- **SMOTE**: Applied for oversampling to balance the dataset.

This README provides an overview of the project structure, instructions to run the pipeline, key findings from EDA, preprocessing steps, feature engineering steps, and model evaluation plans.
