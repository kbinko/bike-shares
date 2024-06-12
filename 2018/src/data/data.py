import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pandas.api.types import CategoricalDtype
from matplotlib.ticker import FuncFormatter

# Importing Data from Capital Bikeshare

jan = pd.read_csv("../../data/raw/201801_capitalbikeshare_tripdata.csv")
feb = pd.read_csv("../../data/raw/201802-capitalbikeshare-tripdata.csv")
mar = pd.read_csv("../../data/raw/201803-capitalbikeshare-tripdata.csv")
apr = pd.read_csv("../../data/raw/201804-capitalbikeshare-tripdata.csv")
may = pd.read_csv("../../data/raw/201805-capitalbikeshare-tripdata.csv")
jun = pd.read_csv("../../data/raw/201806-capitalbikeshare-tripdata.csv")
jul = pd.read_csv("../../data/raw/201807-capitalbikeshare-tripdata.csv")
aug = pd.read_csv("../../data/raw/201808-capitalbikeshare-tripdata.csv")
sep = pd.read_csv("../../data/raw/201809-capitalbikeshare-tripdata.csv")
oct = pd.read_csv("../../data/raw/201810-capitalbikeshare-tripdata.csv")
nov = pd.read_csv("../../data/raw/201811-capitalbikeshare-tripdata.csv")
dec = pd.read_csv("../../data/raw/201812-capitalbikeshare-tripdata.csv")

# Concatenating all the data into one dataframe and saving it as a csv file

y2018 = pd.concat([jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec])
y2018.shape
y2018.to_csv("../../data/raw/2018.csv")

mean_duration = y2018.Duration.mean() / 60  # Average duration of a trip in minutes

# Top 20 most popular trips by station name
sum_column = y2018["Start station"] + " --> " + y2018["End station"]
sum_column.value_counts().head(20)

# Checking data consistency by checking top 20 most popular trips by station number
y2018["Start station number"] = y2018["Start station number"].astype(str)
y2018["End station number"] = y2018["End station number"].astype(str)
sum_column = y2018["Start station number"] + " --> " + y2018["End station number"]
sum_column.value_counts().head(20)

# Checking the number of unique bikes in the dataset
y2018["Bike number"].nunique()

# 20 most popular bikes
y2018["Bike number"].value_counts().head(20)

# saving bikes that have been used less than 50 times
bike_counts = y2018["Bike number"].value_counts()
less_than_50 = bike_counts[bike_counts < 50]

# top 10 end stations
top_end = y2018["End station"].value_counts().head(10)

# top 10 start stations
top_start = y2018["Start station"].value_counts().head(10)

plt_styles = [
    "Solarize_Light2",
    "_classic_test_patch",
    "_mpl-gallery",
    "_mpl-gallery-nogrid",
    "bmh",
    "classic",
    "dark_background",
    "fast",
    "fivethirtyeight",
    "ggplot",
    "grayscale",
    "seaborn-v0_8",
    "seaborn-v0_8-bright",
    "seaborn-v0_8-colorblind",
    "seaborn-v0_8-dark",
    "seaborn-v0_8-dark-palette",
    "seaborn-v0_8-darkgrid",
    "seaborn-v0_8-deep",
    "seaborn-v0_8-muted",
    "seaborn-v0_8-notebook",
    "seaborn-v0_8-paper",
    "seaborn-v0_8-pastel",
    "seaborn-v0_8-poster",
    "seaborn-v0_8-talk",
    "seaborn-v0_8-ticks",
    "seaborn-v0_8-white",
    "seaborn-v0_8-whitegrid",
    "tableau-colorblind10",
]

# visualizing the top 10 start stations and saving the plot

plt.figure(figsize=(18, 8), dpi=120)
ax = top_start.plot.barh()
plt.style.use("seaborn-v0_8-darkgrid")
plt.title("Top 10 Start Stations")
plt.tight_layout()
plt.savefig("../../reports/figures/top_start_stations.png", dpi=120)
plt.show()

# converting the start and end time to datetime
y2018["Start date"] = pd.to_datetime(y2018["Start date"], errors="ignore")
y2018["End date"] = pd.to_datetime(y2018["End date"], errors="ignore")
y2018.dtypes

# Creating new columns for the day of the week
y2018["Day of the week"] = y2018["Start date"].dt.day_name()

# counts of rides per day of the week
day_of_week = y2018["Day of the week"].value_counts()

# changing the order of the days of the week
weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_of_week = day_of_week.reindex(weekday)

# visualizing the number of rides per day of the week

plt.figure(figsize=(18, 8), dpi=120)
ax = day_of_week.plot.bar()
plt.style.use("seaborn-v0_8-poster")
plt.title("Rides per day of the week")
plt.ylabel("Number of rides")
plt.tight_layout()
plt.savefig("../../reports/figures/rides_per_day.png", dpi=120)
plt.show()


# Adding the number of days in the year to the day of the week dataframe
day_of_week2 = day_of_week.to_frame()
day_of_week2["count_of_days"] = [53, 52, 52, 52, 52, 52, 52]

# Calculating the average number of rides per day of the week
day_of_week2["average_per_day"] = round(
    day_of_week2["count"] / day_of_week2["count_of_days"]
)

# visualizing the average number of rides per day of the week
plt.figure(figsize=(18, 8), dpi=120)
ax = day_of_week2["average_per_day"].plot.bar()
plt.style.use("seaborn-v0_8-poster")
plt.title("Average rides per day of the week")
plt.ylabel("Average number of rides")
plt.tight_layout()
plt.savefig("../../reports/figures/average_rides_per_day.png", dpi=120)
plt.show()

# Average duration of rides per day of the week
avg_duration = round(y2018.groupby("Day of the week")["Duration"].mean() / 60, 2)
avg_duration = avg_duration.reindex(weekday)

# Total duration of rides per day of the week
total_duration = round(y2018.groupby("Day of the week")["Duration"].sum() / 60, 2)
total_duration = total_duration.reindex(weekday)

# Adding total duration to the day of the week dataframe
day_of_week2["tot_dur_minute"] = total_duration

# Adding average duration to the day of the week dataframe
day_of_week2["avg_dur_minute"] = avg_duration

# visualizing the total duration of rides per day of the week
plt.figure(figsize=(18, 8), dpi=120)
ax = day_of_week2.plot.barh(y="tot_dur_minute", legend=False)
plt.style.use("seaborn-v0_8-poster")
plt.title("Total duration of rides per weekday")
plt.ylabel("Day")
plt.xlabel("Minutes")
plt.ticklabel_format(style="plain", axis="x")
plt.tight_layout()
plt.savefig("../../reports/figures/total_duration_per_day.png", dpi=120)
plt.show()

# visualizing the average duration of rides per day of the week
plt.figure(figsize=(18, 8), dpi=120)
plt.style.use("fivethirtyeight")
# Plot with the current color map
ax = day_of_week2.plot.barh(
    y="avg_dur_minute", legend=False, cmap="viridis", edgecolor="grey"
)
plt.title(f"Average Duration of Rides per Weekday", fontsize=20)
plt.ylabel("Day", fontsize=15)
plt.xlabel("Minutes", fontsize=15)
plt.ticklabel_format(style="plain", axis="x")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# Adding data labels
for index, value in enumerate(day_of_week2["avg_dur_minute"]):
    plt.text(
        value, index, f"{value:.2f}", ha="left", va="center", fontsize=12, color="black"
    )
plt.tight_layout()
plt.savefig(f"../../reports/figures/average_duration_per_day.png", dpi=120)
plt.show()
