import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append("..")
from utility import plot_settings
from utility.visualize import plot_predicted_vs_true, plot_residuals, regression_scatter

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

bike_data = pd.read_pickle("../../data/processed/bike_data_processed.pkl")
target = "rentals"

# --------------------------------------------------------------
# Train test split
# --------------------------------------------------------------


# --------------------------------------------------------------
# Train model
# --------------------------------------------------------------

# Define preprocessing for numeric columns (scale them)

# Define preprocessing for categorical features (encode them)


# Combine preprocessing steps


# Build the pipeline


# fit the pipeline to train the model on the training set

# --------------------------------------------------------------
# Evaluate the model
# --------------------------------------------------------------

# Get predictions

# Display metrics


# Visualize results

# --------------------------------------------------------------
# Export model
# --------------------------------------------------------------



"""
In Python, you can use joblib or pickle to serialize (and deserialize) an object structure into (and from) a byte stream. 
In other words, it's the process of converting a Python object into a byte stream that can be stored in a file.

https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html

"""

