import pandas as pd

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

# Convert Start date to datetime
df["Start date"] = pd.to_datetime(df["Start date"])


# Extract features from date
df["start_hour"] = df["Start date"].dt.hour
df["start_day_of_week"] = df["Start date"].dt.dayofweek
df["start_month"] = df["Start date"].dt.month
df["start_season"] = df["Start date"].dt.month % 12 // 3 + 1

# Convert member type to binary, member = 1, casual = 0
df["Member type"] = df["Member type"].apply(lambda x: 1 if x == "Member" else 0)

# Droping start/end station names since we have the station ids, bike number is not relevant, also the end station number and end date - since in real-life scenario, when we want to predict the duration of a trip, we won't have this information.
df.drop(
    [
        "Start station",
        "End station",
        "Bike number",
        "End station number",
        "End date",
    ],
    axis=1,
    inplace=True,
)
# Changing Duration to minutes, since that's how a price is calculated
df["Duration"] = df["Duration"] / 60

df.to_pickle("../../data/processed/2018_processed.pkl")
