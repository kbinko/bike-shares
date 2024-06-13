from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = "../../../data/processed/2018_processed.csv"
df = pd.read_csv(file_path)


# List of methods for outlier detection
def detect_outliers(df):
    methods = ["Z-Score", "Isolation Forest", "Local Outlier Factor"]
    for method in methods:
        if method == "Z-Score":
            # Calculate Z-scores
            df["z_score"] = np.abs(stats.zscore(df["Duration"]))
            threshold = 3
            outliers = df[df["z_score"] > threshold]
            label = "Z-Score"

        elif method == "Isolation Forest":
            iso = IsolationForest(contamination=0.01, random_state=42)
            df["outlier_score"] = iso.fit_predict(df[["Duration"]])
            outliers = df[df["outlier_score"] == -1]
            label = "Isolation Forest"

        elif method == "Local Outlier Factor":
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
            df["lof_score"] = lof.fit_predict(df[["Duration"]])
            outliers = df[df["lof_score"] == -1]
            label = "Local Outlier Factor"

        # Sample the data for visualization
        sample_data = df.sample(n=5000, random_state=42)
        sample_outliers = (
            outliers.sample(n=5000, random_state=42)
            if len(outliers) > 5000
            else outliers
        )

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.scatter(
            sample_data.index,
            sample_data["Duration"],
            label="df",
            alpha=0.3,
            s=10,
            color="blue",
        )
        plt.scatter(
            sample_outliers.index,
            sample_outliers["Duration"],
            color="red",
            label="Outliers",
            s=10,
        )
        plt.title(f"Scatter plot of Duration with {label} Outliers")
        plt.xlabel("Index")
        plt.ylabel("Duration")
        plt.legend(loc="upper right")
        plt.show()

        # Output the number of outliers
        print(f"Method: {label}")
        print(f"Number of outliers detected: {len(outliers)}")
        print(f"Percentage of outliers: {len(outliers) / len(df) * 100:.2f}%\n")
