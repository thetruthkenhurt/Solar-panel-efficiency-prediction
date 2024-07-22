#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Print current directory structure
echo "Current directory structure:"
ls -R

python src/data_loader.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_model.py
