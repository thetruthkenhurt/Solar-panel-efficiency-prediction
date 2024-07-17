# AIAP 18 Technical Assessment

## Full Name: [Your Full Name]
## Email: [Your Email]

### Project Structure
- `src/`: Contains all Python scripts for the pipeline
- `run.sh`: Bash script to execute the pipeline
- `requirements.txt`: List of dependencies

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

### Key Findings from EDA
- Higher temperature ranges correlate with lower solar panel efficiency.
- Rainfall intensity has a negative impact on efficiency.
- High humidity and cloud cover reduce efficiency levels.

### Feature Engineering
- Added features such as temperature range and rainfall intensity.
- Binned temperature and rainfall features for better model performance.

### Model Evaluation
- Used Random Forest Classifier for classification.
- Evaluated using classification report, confusion matrix, and ROC AUC score.

### Stratification and Sampling
- Used stratified split to maintain class distribution in training and testing sets.
- Applied SMOTE for oversampling to balance the dataset.
