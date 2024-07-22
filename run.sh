#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if the data directory exists
if [ ! -d "data" ]; then
  echo "Data directory 'data' does not exist."
  exit 1
fi

# Check if the database files exist
if [ ! -f "data/weather.db" ]; then
  echo "Database file 'weather.db' does not exist in 'data' directory."
  exit 1
fi

if [ ! -f "data/air_quality.db" ]; then
  echo "Database file 'air_quality.db' does not exist in 'data' directory."
  exit 1
fi

export DB_PATH=data
python src/data_loader.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_model.py