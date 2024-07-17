#!/bin/bash
export DB_PATH=data
python src/data_loader.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
