import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys
import cloudpickle as cp


sys.path.append("..")
from data.data_preprocessing import process_bike_data
from utility import plot_settings
from utility.visualize import plot_predicted_vs_true, regression_scatter, plot_residuals


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_csv("../../data/raw/daily-bike-share.csv.")


process_bike_data(
    "../../data/raw/daily-bike-share.csv",
    "../../data/processed/bike_data_processed1.pkl",
)
bike_data = pd.read_pickle("../../data/processed/bike_data_processed1.pkl")


# --------------------------------------------------------------
# Load model
# --------------------------------------------------------------

with open("../../models/model.pkl", "rb") as f:
    model = cp.load(f)

# --------------------------------------------------------------
# Make predictions
# --------------------------------------------------------------

X_new = bike_data.drop("cnt", axis=1)
y_new = bike_data["cnt"]
predictions = model.predict(X_new)

# --------------------------------------------------------------
# Evaluate results
# --------------------------------------------------------------

rmse = np.sqrt(mean_squared_error(y_new, predictions))
r2 = r2_score(y_new, predictions)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# Visualize results

plot_predicted_vs_true(y_new, predictions)
