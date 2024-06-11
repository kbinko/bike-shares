import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys

sys.path.append("..")
from utility import plot_settings
from utility.visualize import plot_predicted_vs_true, regression_scatter, plot_residuals


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

bike_data = pd.read_csv("../../data/raw/daily-bike-share.csv")
new_sample = bike_data.sample(20)

# --------------------------------------------------------------
# Load model
# --------------------------------------------------------------


# --------------------------------------------------------------
# Make predictions
# --------------------------------------------------------------

# --------------------------------------------------------------
# Evaluate results
# --------------------------------------------------------------


# Visualize results
