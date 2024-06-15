import pandas as pd
from LearningAlgorithms import evaluate_regression_models, evaluate_feature_sets
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


# Load data

df = pd.read_pickle("../../data/features/2018_features.pkl")
df_outliers = pd.read_pickle("../../data/features/2018_features_outliers.pkl")


# Sampling data for faster training, taking one week from each month to make sure the month differences are captured
def sample_week_from_each_month(df, date_column="Start date", target_column="Duration"):
    df[date_column] = pd.to_datetime(df[date_column])
    sampled_df = pd.DataFrame()
    for month in range(1, 13):
        monthly_data = df[df[date_column].dt.month == month]
        weekly_sample = monthly_data[
            (monthly_data[date_column].dt.day >= 8)
            & (monthly_data[date_column].dt.day <= 14)
        ]
        sampled_df = pd.concat([sampled_df, weekly_sample])
    return sampled_df


sampled_df = sample_week_from_each_month(df)
sampled_df_outliers = sample_week_from_each_month(df_outliers)
# Now dropping start date column since we needed it only to sample the data
sampled_df.drop("Start date", axis=1, inplace=True)
sampled_df_outliers.drop("Start date", axis=1, inplace=True)

# The dataset has few duplicated columns, we will drop them
sampled_df = sampled_df.loc[:, ~sampled_df.columns.duplicated()]
sampled_df_outliers = sampled_df_outliers.loc[
    :, ~sampled_df_outliers.columns.duplicated()
]


# Choosing feature sets by slicing the dataframe columns

feature_set1 = list(sampled_df.columns)[1:8] + ["Start station number_freq"]
feature_set2 = feature_set1 + [
    "station_mean_duration",
    "station_median_duration",
    "station_trip_count",
]

feature_set3 = feature_set2 + [
    "day_part_encoded",
    "member_day_interaction_encoded",
    "member_hour_interaction_encoded",
    "member_day_part_interaction_encoded",
    "member_season_interaction_encoded",
]

feature_set4 = feature_set3 + ["pca_0", "pca_1", "pca_2", "pca_3", "pca_4"]

feature_set5 = feature_set4 + [
    "start_hour^2",
    "start_hour start_day_of_week",
    "start_hour start_month",
    "start_hour start_season",
    "start_day_of_week^2",
    "start_day_of_week start_month",
    "start_day_of_week start_season",
    "start_month^2",
    "start_month start_season",
    "start_season^2",
]

feature_pca = ["pca_0", "pca_1", "pca_2", "pca_3", "pca_4"]

feature_sets = [
    feature_set1,
    feature_set2,
    feature_set3,
    feature_set4,
    feature_set5,
    feature_pca,
]
"""
 This is the standard procedure that normally i would follow, unfortunately, the dataset is too big to be trained on my local machine, checking one feature set [feature_set1] on NN (without even evaluating regression models) didn't even finish after 1 hour.


# Evaluating the regression models, given the big size of the dataset, we will only use 2 models that i think are the most effective for this task
# models_to_evaluate = ["RandomForest", "NeuralNetwork"]
# Evaluating the models with the outliers
# results, best_model = evaluate_regression_models(
#     sampled_df, target_column="Duration", models_to_evaluate=models_to_evaluate
# )
# Evaluating the models without the outliers
# results_outliers, best_model_outliers = evaluate_regression_models(
#     sampled_df_outliers, target_column="Duration", models_to_evaluate=models_to_evaluate
# )
# Evaluating the feature sets using neural network
# results, best_feature_set, best_model = evaluate_feature_sets(
#     sampled_df, target_column="Duration", feature_sets=feature_sets
# )

 Neural Network
 nn = MLPRegressor(
     hidden_layer_sizes=(100, 100),
     activation="relu",
     solver="adam",
     alpha=0.0001,
     batch_size="auto",
     learning_rate="constant",
     learning_rate_init=0.001,
     max_iter=200,
     shuffle=True,
     random_state=42,
     tol=0.0001,
     verbose=False,
     warm_start=False,
     momentum=0.9,
     nesterovs_momentum=True,
     early_stopping=False,
     validation_fraction=0.1,
     beta_1=0.9,
     beta_2=0.999,
     epsilon=1e-08,
     n_iter_no_change=10,
     max_fun=15000,
 )
 for X, y, name in zip(
     [X_train, X_train_pca], [y_train, y_train_pca], ["feature_set1", "pca"]
 ):
     nn.fit(X, y)
     print(f"Training {name} done")
     print(f"Training score {nn.score(X, y)}")
     print(f"Testing score {nn.score(X_test, y_test)}")
     print("--------------------------------------------------")
 Unfortunately my local machine can't handle complex models with this big dataset, so i will just test simpler ones and see the results. Im pretty sure that fully trained NN or RandomForest with best parameters and feature set will give the best results, sadly this is not possible.

"""
# spliting the data into train and test
# feature set 1 and pca (shortest ones, hope they capture the most important features)
X = sampled_df.drop("Duration", axis=1)
X_pca = sampled_df[feature_pca + ["Duration"]]
y = sampled_df["Duration"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# Checking best parameters for decision tree since it performed best. Still not the best way to do it, but it's the only way on my local machine

models_to_evaluate = ["DecisionTree"]
results, best_model = evaluate_regression_models(
    X_pca, target_column="Duration", models_to_evaluate=models_to_evaluate
)
