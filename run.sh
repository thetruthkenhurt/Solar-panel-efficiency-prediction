#!/bin/bash


# Exit immediately if a command exits with a non-zero status.
set -e

export DB_PATH=data
python src/data_loader.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_model.py

#!/bin/bash


