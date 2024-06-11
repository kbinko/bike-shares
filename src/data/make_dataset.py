import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

"""
The data used in this exercise is derived from Capital Bikeshare 
and is used in accordance with the published license agreement.
https://www.capitalbikeshare.com/system-data

"""


bike_data = pd.read_csv("../../data/raw/daily-bike-share.csv")
bike_data.info()

