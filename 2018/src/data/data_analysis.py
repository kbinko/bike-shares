import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "../../data/raw/2018.csv"
df = pd.read_csv(file_path)
df.drop(["Unnamed: 0"], axis=1, inplace=True)

# Check missing values
missing = df.isnull().sum()  # No missing values

# Check data types
df.dtypes

# Convert Start date and End date to datetime
df["Start date"] = pd.to_datetime(df["Start date"])
df["End date"] = pd.to_datetime(df["End date"])

# Extract features from dates
