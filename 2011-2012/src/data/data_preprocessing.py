import pandas as pd

"""
The data used in this exercise is derived from Capital Bikeshare 
and is used in accordance with the published license agreement.
https://www.capitalbikeshare.com/system-data

"""


def process_bike_data(input_path, output_path):
    # Load data
    bike_data = pd.read_csv(input_path)
    bike_data = bike_data.drop(["registered", "casual"], axis=1)

    # Select features that are relevant for the analysis
    relevant_features = [
        "season",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "cnt",
    ]
    bike_data = bike_data[relevant_features]

    # Convert non-numerical columns to dtype category
    categorical_features = [
        "season",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    bike_data[categorical_features] = bike_data[categorical_features].astype("category")

    # Interaction features
    bike_data["temp_hum"] = bike_data["temp"] * bike_data["hum"]
    bike_data["temp_windspeed"] = bike_data["temp"] * bike_data["windspeed"]
    bike_data["atemp_hum"] = bike_data["atemp"] * bike_data["hum"]
    bike_data["atemp_windspeed"] = bike_data["atemp"] * bike_data["windspeed"]

    # Binning temperature into categories
    bike_data["temp_bin"] = pd.cut(
        bike_data["temp"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["cold", "mild", "warm", "hot", "very hot"],
    )

    # Lagged features for rentals
    bike_data["cnt_lag_1"] = bike_data["cnt"].shift(1)
    bike_data["cnt_lag_2"] = bike_data["cnt"].shift(2)

    # Rolling mean for cnt
    bike_data["cnt_roll_mean_3"] = bike_data["cnt"].rolling(window=3).mean()

    # Drop NaN values from lag features
    bike_data = bike_data.dropna()

    # Export the cleaned data
    bike_data.to_pickle(output_path)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    process_bike_data(
        "../../data/raw/daily-bike-share.csv",
        "../../data/processed/bike_data_processed.pkl",
    )
