import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from feature_evaluation import target_encode, select_important_features
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from outliers import detect_outliers

# Load the data
file_path = "../../../data/processed/2018_processed.pkl"
df = pd.read_pickle(file_path)

# Sampling the data
# df = df.sample(frac=0.1, random_state=42)

# -------------------------------------
# Temporal features
# -------------------------------------


# Assigning day part
def assign_day_part(hour):
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


# Assigning rush hour
def is_rush_hour(hour):
    if 6 <= hour < 9 or 15 <= hour < 19:
        return 1
    else:
        return 0


df["day_part"] = df["start_hour"].apply(assign_day_part)
df["is_rush_hour"] = df["start_hour"].apply(is_rush_hour)
df["is_weekend"] = df["start_day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

# -------------------------------------
# Interaction features
# -------------------------------------

df["member_day_interaction"] = (
    df["Member type"].astype(str) + "_" + df["start_day_of_week"].astype(str)
)
df["member_hour_interaction"] = (
    df["Member type"].astype(str) + "_" + df["start_hour"].astype(str)
)
df["member_day_part_interaction"] = (
    df["Member type"].astype(str) + "_" + df["day_part"].astype(str)
)
df["member_season_interaction"] = (
    df["Member type"].astype(str) + "_" + df["start_season"].astype(str)
)

# -------------------------------------
# Aggregated features
# -------------------------------------

station_mean_duration = (
    df.groupby("Start station number")["Duration"]
    .mean()
    .rename("station_mean_duration")
)
station_median_duration = (
    df.groupby("Start station number")["Duration"]
    .median()
    .rename("station_median_duration")
)
station_trip_count = (
    df.groupby("Start station number").size().rename("station_trip_count")
)

df = df.merge(station_mean_duration, on="Start station number", how="left")
df = df.merge(station_median_duration, on="Start station number", how="left")
df = df.merge(station_trip_count, on="Start station number", how="left")

# -------------------------------------
# Target encoding categorical features
# -------------------------------------

columns_to_encode = [
    "day_part",
    "member_day_interaction",
    "member_hour_interaction",
    "member_day_part_interaction",
    "member_season_interaction",
]

df = target_encode(df, "Duration", columns_to_encode)
df.drop(columns_to_encode, axis=1, inplace=True)


# Frequency encoding start station number - this is a categorical feature but with too much unique values to one hot encode
def frequency_encoding(df, column):
    freq = df[column].value_counts(normalize=True)
    df[f"{column}_freq"] = df[column].map(freq)
    return df


df = frequency_encoding(df, "Start station number")
df.drop("Start station number", axis=1, inplace=True)

# -------------------------------------
# Scaling numerical features
# -------------------------------------

scaler = StandardScaler()
numerical_features = [
    "start_hour",
    "start_day_of_week",
    "start_month",
    "start_season",
    "day_part_encoded",
    "member_day_interaction_encoded",
    "member_hour_interaction_encoded",
    "member_day_part_interaction_encoded",
    "member_season_interaction_encoded",
    "Start station number_freq",
]

df[numerical_features] = scaler.fit_transform(df[numerical_features])

# -------------------------------------
# Applying PCA and Polynomial features
# -------------------------------------

# Applying PCA and adding the components to the dataframe
df_pca = df[numerical_features]
pca = PCA(n_components=5)
pca_components = pca.fit_transform(df_pca)

for i in range(pca_components.shape[1]):
    df[f"pca_{i}"] = pca_components[:, i]

df.shape[0]
# Adding Polynomial features to the dataframe
poly_features = ["start_hour", "start_day_of_week", "start_month", "start_season"]
poly = PolynomialFeatures(2, include_bias=False)
poly_features_transformed = poly.fit_transform(df[poly_features])
poly_feature_names = poly.get_feature_names_out(poly_features)
df_poly = pd.DataFrame(
    poly_features_transformed, columns=poly_feature_names, index=df.index
)
df = pd.concat([df, df_poly], axis=1)

# -------------------------------------
# Outliers
# -------------------------------------

# Checking which outliers detection method is the most effective
detect_outliers(df, "Duration", df.shape[0])

# Using Isolation Forest to detect outliers
iso = IsolationForest(contamination=0.01, random_state=42)
df["outlier_score"] = iso.fit_predict(df[["Duration"]])
outliers = df[df["outlier_score"] == -1]

# Removing the outliers and exporting the dataset
df_outliers_removed = df.copy()
df_outliers_removed.drop(outliers.index, inplace=True)
df_outliers_removed.to_pickle("../../../data/processed/2018_processed_outliers.pkl")

# -------------------------------------
# Selecting the most important features
# -------------------------------------
# Sampling the data to speed up the feature selection process
# df = df.sample(frac=0.1, random_state=42)
df["Duration"].min()
