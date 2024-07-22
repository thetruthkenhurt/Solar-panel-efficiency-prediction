# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier

# Train Logistic Regression Model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Train Random Forest Model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Train XGBoost Model
def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# Train Ensemble Model using a mixture of above models
def train_ensemble(X_train, y_train):
    # Define individual models
    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Create ensemble model
    ensemble_model = VotingClassifier(estimators=[
        ('lr', log_reg), 
        ('rf', rf), 
        ('xgb', xgb)], voting='soft')

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

# To output classification report and evaluation metrics
def evaluate_model(model, X_test, y_test, model_type='single'):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if model_type != 'ensemble' else model.predict_proba(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))

# Function to train and evaluate models without tuning
def train_and_evaluate(X, y, model_type='logistic_regression'):
    # Stratified split to maintain class distribution in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Oversampling using SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train the selected model
    if model_type == 'logistic_regression':
        model = train_logistic_regression(X_resampled, y_resampled)
    elif model_type == 'random_forest':
        model = train_random_forest(X_resampled, y_resampled)
    elif model_type == 'xgboost':
        model = train_xgboost(X_resampled, y_resampled)
    elif model_type == 'ensemble':
        model = train_ensemble(X_resampled, y_resampled)
    else:
        raise ValueError("Invalid model_type. Expected one of: 'logistic_regression', 'random_forest', 'xgboost', 'ensemble'")

    evaluate_model(model, X_test, y_test)
    return model

def perform_grid_search(model, param_grid, X_train, y_train, scoring='accuracy', cv=3, n_jobs=-1):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Function for hyperparameter tuning and training + evaluation of tuned models
def hyperparameter_tuning_and_evaluate_grid(X, y, param_grid, model_type='logistic_regression', use_subset=True, subset_size=0.1):
    # If use_subset is True, use a subset of the data
    if use_subset:
        X_subset, _, y_subset, _ = train_test_split(X, y, train_size=subset_size, stratify=y, random_state=42)
    else:
        X_subset, y_subset = X, y
    
    # Stratified split to maintain class distribution in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, stratify=y_subset, random_state=42)

    # Oversampling using SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Perform grid search and train the selected model
    if model_type == 'logistic_regression':
        best_model, best_params = perform_grid_search(LogisticRegression(max_iter=1000), param_grid, X_resampled, y_resampled)
    elif model_type == 'random_forest':
        best_model, best_params = perform_grid_search(RandomForestClassifier(), param_grid, X_resampled, y_resampled)
    elif model_type == 'xgboost':
        best_model, best_params = perform_grid_search(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, X_resampled, y_resampled)
    else:
        raise ValueError("Invalid model_type. Expected one of: 'logistic_regression', 'random_forest', 'xgboost'")
    
    # Cross-validation using Stratified K-Folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=skf, scoring='accuracy')
    print(f'Stratified K-Fold Cross-Validation Scores for {model_type}: {cv_scores}')
    print(f'Mean CV Score for {model_type}: {cv_scores.mean()}')

    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)
    return best_model, best_params

# Train + evaluate the models with the best parameters obtained from subset training
def train_with_best_params(X, y, model_type='logistic_regression', best_params=None):
    # Stratified split to maintain class distribution in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Oversampling using SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train the selected model with best parameters
    if model_type == 'logistic_regression':
        model = LogisticRegression(**best_params, max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**best_params)
    elif model_type == 'xgboost':
        model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("Invalid model_type. Expected one of: 'logistic_regression', 'random_forest', 'xgboost'")

    model.fit(X_resampled, y_resampled)
    

 # Cross-validation using Stratified K-Folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='accuracy')
    print(f'Stratified K-Fold Cross-Validation Scores for {model_type}: {cv_scores}')
    print(f'Mean CV Score for {model_type}: {cv_scores.mean()}')

    # Train the model on the full training data
    model.fit(X_resampled, y_resampled)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    return model
