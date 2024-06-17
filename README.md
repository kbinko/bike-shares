# Capital Bikeshare project & Data Analysis

## Table of contents

* [Project Introduction](https://github.com/kbinko/bike-shares/edit/main/README.md#project-introduction)
* [2011-2012 shares count model prediction](https://github.com/kbinko/bike-shares/edit/main/README.md#2011-2012-shares-count-model-prediction)
* [2018 exploratory analysis and model building]

## Project Introduction

The data was pulled from Capital Bikeshare, a company that offers a bicycle sharing system in Washington D.C
The goal of the project was to create predictive models and analyze the data to find interesting behaviours and patterns. Based on my finds, I created different plots showing various interactions. 
With the data from 2011-2012 i was able to create a Neural Network model that was able to predict shares count with an average error of 3-4%. I tried to create a model to predict trip duration with the data from 2018, but unfortunately my old machine was not able to procces this large data - after preprocessing I ended up with two datasets, one 3542684 rows x 38 columns, and the other with 3507599 rows x 38 columns. 

## 2011-2012 shares count model prediction

Originally I had this dataset : 

![image](https://github.com/kbinko/bike-shares/assets/59764236/6ded01ab-636c-409e-8fa8-be46bb4106e7)

I dropped casual/registered columns since they were not relevant in this case scenario, the goal was to predict all shares, not split by member types. 
I added few features, like temperature * humidity, temperature * windspeed etc. I realized that they are important, beacuse for example high temperature is bearable with low humidity/big windspeed. I also added "rentals_lag" which were the ammount of rentals yesterday/dwo days ago + mean ammount of rentals from 3 days. Thanks to this data preprocessing and feature engineering steps i was able to vastly increase model performance. The model I used was KerasRegressor. I tested it along with xgboost and RandomForestRegressor and found out that it was performing best. I created a pipeline where I added Polynomial Features with a degree of 2 and scaled numerical features using StandardScaler. Categorical features were encoded using OneHotEncoder. For the model, I initially used RandomizedSearchCV to find the best parameters. This is the list of parameters checked: 
```
param_grid = {
    "regressor__model__num_neurons": [64, 128, 256],
    "regressor__model__num_layers": [2, 3, 4],
    "regressor__model__dropout_rate": [0.2, 0.3, 0.4],
    "regressor__epochs": [50, 100],
    "regressor__batch_size": [16, 32, 64],
```

Then I created the model using best parameters :
- neurons: 256
- layers: 2
- dropout rate: 0.2
- epochs: 100
- batch size: 16

Using this model I was able to achieve mean squared error of 140-190 and R^2 score of 0.992 - 0.996
With averege ammount of shares being 4500, it means that my model had an average error of 3-4%, which in this case is quite fenomenal. 
Without data preprocessing and feature engineering, model had an MSE ranging from 500 - 700, which shows that I have handled this case well.

Here are two plots from this part of the project, model's learning curve and predicted vs true:

![image](https://github.com/kbinko/bike-shares/assets/59764236/4cbe7c2f-5bd6-4b9d-bed8-cd4ce2d01efd)
![image](https://github.com/kbinko/bike-shares/assets/59764236/5b27bd49-2abd-4f30-8f0c-982277b1c5d8)


## 2018 exploratory analysis and model building
### Overview
This report presents a comprehensive analysis of bike-sharing data for the year 2018, sourced from Capital Bikeshare. The analysis includes various aspects such as hourly and monthly distribution, ride duration, seasonal variation, and station popularity. The insights derived from this analysis can help in understanding user behavior and optimizing the bike-sharing system. I planned to create a model that could predict trip duration using this data, but unfortunately the dataset was too big for my 9 years old machine. 

This is how raw dataset looked like:

![image](https://github.com/kbinko/bike-shares/assets/59764236/4eb8d076-9c8f-4a8b-b12e-b68da1ae996e)

Originally there were 12 datasets for each month, I combined them into one and then deleted them to save space.
I focused on finding different interactions that were meaningfull and provided insight into User's behaviour. 

### Key Findings
1. Hourly Distribution of Bike Rentals by Member Type
   
![image](https://github.com/kbinko/bike-shares/assets/59764236/5876dbb4-c434-409e-b0c4-9cfbad6fbbfd)

  * **Members** show clear peaks during commute hours (around 8 AM and 5-6 PM), indicating they primarily use bikes for commuting.
  * **Casual** Users have a more steady usage pattern throughout the day, with a slight increase during the afternoon.
  * The difference in peaks shows that the company probably has very viable member perks - user's who use bikes to commute to/from work probably decide to become a member
    
2. Count of Rides per Minute Duration
   
![image](https://github.com/kbinko/bike-shares/assets/59764236/151d7a1f-242b-4867-b5cb-924611c2c02e)
![image](https://github.com/kbinko/bike-shares/assets/59764236/d8390aeb-8e22-478f-a172-3f673f25082b)

  * **Short rides predominate:** Most rides are short, with a steep drop off after 10 minutes.

3. Monthly Distribution of Bike Rentals by Member Type

![image](https://github.com/kbinko/bike-shares/assets/59764236/97ea1193-1ffd-4817-b700-d3ff5f2a6cc8)


* **Seasonal Variation:** Both member and casual rides peak during the summer months.
* **Higher Member Usage:** Members consistently have more rides compared to casual users throughout the year.
* **Difference in fall season:** Throughout the whole year members and casual users behave simillary, with the difference being 10th month, where the member's activity increases.

4. Season-wise Hourly Distribution of Bike Rentals

![image](https://github.com/kbinko/bike-shares/assets/59764236/880e803d-0ec1-4c6c-be43-12bd63dffa94)

* **Consistent Patterns:** Morning and evening peaks are consistent across seasons, with more pronounced peaks in warmer seasons.

5. Total Duration of Rides per Weekday

![image](https://github.com/kbinko/bike-shares/assets/59764236/76f6fd44-aa9f-4ef8-b735-75e741aeb1e2)


* **Weekend Peaks:** Saturday and Sunday have the highest total ride durations, indicating more leisure riding on weekends.

6. Rides per Weekday

![image](https://github.com/kbinko/bike-shares/assets/59764236/882f35c3-ab52-420c-8dae-02f67529e2b1)

* **Consisten Weekday Rides:** The number of rides is fairly consistent during the week, with slight drop on weekends

7. Average Rides Duration per Starting Hour

![image](https://github.com/kbinko/bike-shares/assets/59764236/0fa8faa8-4cf3-42b4-8c41-bdd4ab45359c)

* **Early Morning Rides:** Rides starting early in the morning tend to be longer.
* **Short Commute Rides:** Commute hours have shorter average durations, suggesting short commutes.
