from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt


def detect_outliers(df, column, sample_size):
    methods = ["Z-Score", "Isolation Forest", "Local Outlier Factor"]
    for method in methods:
        if method == "Z-Score":
            # Calculate Z-scores
            df["z_score"] = np.abs(stats.zscore(df[column]))
            threshold = 3
            outliers = df[df["z_score"] > threshold]
            label = "Z-Score"

        elif method == "Isolation Forest":
            iso = IsolationForest(contamination=0.01, random_state=42)
            df["outlier_score"] = iso.fit_predict(df[[column]])
            outliers = df[df["outlier_score"] == -1]
            label = "Isolation Forest"

        elif method == "Local Outlier Factor":
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
            df["lof_score"] = lof.fit_predict(df[[column]])
            outliers = df[df["lof_score"] == -1]
            label = "Local Outlier Factor"

        # Sample the data for visualization
        sample_data = df.sample(n=sample_size, random_state=42)
        sample_outliers = (
            outliers.sample(n=sample_size, random_state=42)
            if len(outliers) > sample_size
            else outliers
        )

        # Visualization
        plt.figure(figsize=(15, 10), dpi=120)
        plt.scatter(
            sample_data.index,
            sample_data[column],
            label="df",
            alpha=0.3,
            s=10,
            color="blue",
        )
        plt.scatter(
            sample_outliers.index,
            sample_outliers[column],
            color="red",
            label="Outliers",
            s=10,
        )
        plt.title(f"Scatter plot of {column} with {label} Outliers")
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.legend(loc="upper right")
        plt.show()

        # Output the number of outliers
        print(f"Method: {label}")
        print(f"Number of outliers detected: {len(outliers)}")
        print(f"Percentage of outliers: {len(outliers) / len(df) * 100:.2f}%\n")
