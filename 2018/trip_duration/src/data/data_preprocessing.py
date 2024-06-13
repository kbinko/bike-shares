import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load the data
file_path = "../../../data/raw/2018.csv"
df = pd.read_csv(file_path)
df.drop(["Unnamed: 0"], axis=1, inplace=True)
df["Duration"].max()

# Sampling data in case that data takes too long to process
# sampled_data = df.sample(frac=0.1, random_state=42)
# sampled_data.to_csv("../../../data/raw/2018_sampled.csv")

# Check missing values
missing = df.isnull().sum()  # No missing values

# Check data types
df.dtypes

# Convert Start date and End date to datetime
df["Start date"] = pd.to_datetime(df["Start date"])
df["End date"] = pd.to_datetime(df["End date"])

# Extract features from dates
df["start_hour"] = df["Start date"].dt.hour
df["start_day_of_week"] = df["Start date"].dt.dayofweek
df["start_month"] = df["Start date"].dt.month
df["start_season"] = df["Start date"].dt.month % 12 // 3 + 1

# Extract features from End date
df["end_hour"] = df["End date"].dt.hour

# We don't need the original dates anymore
df.drop(["Start date", "End date"], axis=1, inplace=True)

# Convert member type to binary, member = 1, casual = 0
df["Member type"] = df["Member type"].apply(lambda x: 1 if x == "Member" else 0)

# Droping start/end station names since we have the station ids, and bike number since its not really relevant
df.drop(["Start station", "End station", "Bike number"], axis=1, inplace=True)


# Select numerical features to scale
numerical_features = [
    "Duration",
    "Start station number",
    "End station number",
    "start_hour",
    "start_day_of_week",
    "start_month",
    "start_season",
    "end_hour",
]

# Scaling numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])


df.to_pickle("../../../data/processed/2018_processed.pkl")
