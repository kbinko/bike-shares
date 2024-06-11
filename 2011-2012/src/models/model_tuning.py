import sys

import joblib
import numpy as np
import pandas as pd
import cloudpickle as cp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

sys.path.append("..")
from utility import plot_settings
from utility.visualize import plot_predicted_vs_true, plot_residuals, regression_scatter

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

bike_data = pd.read_pickle("../../data/processed/bike_data_processed.pkl")
target = "rentals"

# --------------------------------------------------------------
# Train test split
# --------------------------------------------------------------

X = bike_data.drop(target, axis=1)
y = bike_data[target]

# check the average number of rentals per day

y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------

# Define preprocessing for numeric columns (scale them)

numeric_features = X.select_dtypes(include=["float", "int"]).columns
numeric_transformer = Pipeline(
    steps=[
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
    ]
)

# Define preprocessing for categorical features (encode them)

categorical_features = X.select_dtypes(include=["category", "object"]).columns
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combine preprocessing steps

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# --------------------------------------------------------------
# Training models (XGBoost, Random Forest, Stacking)
# --------------------------------------------------------------


# Define the base models
base_models = [
    ("rf", RandomForestRegressor(n_estimators=100)),
    ("xgb", XGBRegressor(objective="reg:squarederror")),
]

# Define the stacking model
stacked_model = StackingRegressor(
    estimators=base_models, final_estimator=RandomForestRegressor(n_estimators=100)
)

# Build the pipeline

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", stacked_model)]
)
# Define the parameter grid for RandomizedSearchCV

param_grid = {
    "regressor__rf__n_estimators": [100, 200],
    "regressor__rf__max_depth": [None, 10],
    "regressor__xgb__n_estimators": [100, 200],
    "regressor__xgb__max_depth": [3, 5],
    "regressor__xgb__learning_rate": [0.01, 0.1],
    "regressor__final_estimator__n_estimators": [100, 200],
    "regressor__final_estimator__max_depth": [None, 10],
}

# Initialize RandomizerSearchCV
random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_grid, n_iter=50, cv=3, n_jobs=-1, verbose=1
)

# Fit the model
best_model = random_search.fit(X_train, y_train)


# --------------------------------------------------------------
#  Neural Network
# --------------------------------------------------------------


# Define the Keras model


X_train_transformed = preprocessor.fit_transform(X_train)


def build_keras_model(input_dim, num_neurons=256, num_layers=2, dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=input_dim, activation="relu"))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation="relu"))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop", loss="mean_squared_error")
    return model


input_dim = X_train_transformed.shape[1]

# Wrap the Keras model using KerasRegressor
keras_regressor = KerasRegressor(
    model=build_keras_model,
    model__input_dim=input_dim,
    model__num_neurons=256,
    model__num_layers=2,
    model__dropout_rate=0.2,
    optimizer="rmsprop",
    epochs=100,
    batch_size=16,
    verbose=0,
)

# Build the pipeline
pipeline_nn = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", keras_regressor),
    ]
)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    "regressor__model__num_neurons": [64, 128, 256],
    "regressor__model__num_layers": [2, 3, 4],
    "regressor__model__dropout_rate": [0.2, 0.3, 0.4],
    "regressor__epochs": [50, 100],
    "regressor__batch_size": [16, 32, 64],
    "regressor__model__l2_lambda": [0.001, 0.01, 0.1],  # L2 regularization parameter
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline_nn,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    verbose=10,
    n_jobs=-1,
    random_state=42,
)

# Fit the model
best_model = random_search.fit(X_train, y_train)


# Get predictions
predictions_nn = best_model.predict(X_test)

# best model (saved for the future use to not run the model again)
# # KerasRegressor(
# 	model=<function build_keras_model at 0x71ce76139990>
# 	build_fn=None
# 	warm_start=False
# 	random_state=None
# 	optimizer=rmsprop
# 	loss=None
# 	metrics=None
# 	batch_size=16
# 	validation_batch_size=None
# 	verbose=0
# 	callbacks=None
# 	validation_split=0.0
# 	shuffle=True
# 	run_eagerly=False
# 	epochs=100
# 	model__input_dim=175
# 	model__num_neurons=256
# 	model__num_layers=2
# 	model__dropout_rate=0.2
model = pipeline_nn.fit(X_train, y_train)
predictions_nn = pipeline_nn.predict(X_test)


# --------------------------------------------------------------
# Evaluate the model
# --------------------------------------------------------------

# Get predictions
predictions = best_model.predict(X_test)

predictions_nn = model.predict(X_test)

# Display metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Stacked Model:")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

rmse_nn = np.sqrt(mean_squared_error(y_test, predictions_nn))
r2_nn = r2_score(y_test, predictions_nn)

print("Neural Network:")
print(f"RMSE: {rmse_nn}")
print(f"R^2: {r2_nn}")


# Visualize results

plot_predicted_vs_true(y_test, predictions_nn)
regression_scatter(y_test, predictions_nn)
plot_residuals(y_test, predictions_nn, bins=15)

# --------------------------------------------------------------
# Learning curve for NN model
# --------------------------------------------------------------


def plot_learning_curves(model, X, y, cv=3, scoring="neg_mean_squared_error"):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(14, 7))
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


plot_learning_curves(pipeline_nn, X_train, y_train)


# --------------------------------------------------------------
# Export model
# --------------------------------------------------------------


"""
In Python, you can use joblib or pickle to serialize (and deserialize) an object structure into (and from) a byte stream. 
In other words, it's the process of converting a Python object into a byte stream that can be stored in a file.

https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html

"""

ref_cols = list(X.columns)

joblib.dump(value=[pipeline_nn, ref_cols, target], filename="../../models/model.pkl")

with open("../../models/model.pkl", "wb") as f:
    cp.dump(pipeline_nn, f)
