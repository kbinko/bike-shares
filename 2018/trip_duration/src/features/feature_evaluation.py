import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split


def target_encode(df, target, columns, n_splits=5):
    """
    Perform target encoding with cross-validation to prevent data leakage.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    target (str): The name of the target column.
    columns (list): List of categorical columns to encode.
    n_splits (int): Number of splits for cross-validation.

    Returns:
    pd.DataFrame: Dataframe with target encoded features.
    """
    df_encoded = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in columns:
        df_encoded[f"{col}_encoded"] = 0
        for train_idx, val_idx in kf.split(df):
            train, val = df.iloc[train_idx], df.iloc[val_idx]
            means = train.groupby(col)[target].mean()
            df_encoded.loc[val_idx, f"{col}_encoded"] = val[col].map(means)

    return df_encoded


# Function to evaluate feature engineering and visualize the results


def select_important_features(
    df, target_column, n_top_features=15, test_size=0.2, random_state=42
):
    """
    Select the most important features from a dataset using a RandomForestRegressor.

    Parameters:
    df (pd.DataFrame): The input dataframe containing features and target variable.
    target_column (str): The name of the target variable column.
    n_top_features (int): The number of top features to select.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The random seed for reproducibility.

    Returns:
    pd.DataFrame: A dataframe containing the top selected features and the target variable.
    """

    # Split the data into features and target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train the RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Select top features
    top_features = X_train.columns[indices][:n_top_features]

    # Print the top features and their importance scores
    print("Top features and their importance scores:")
    for i in range(n_top_features):
        print(
            f"{i + 1}. Feature: {top_features[i]}, Importance: {importances[indices[i]]}"
        )

    # Visualize the feature importances
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {n_top_features} Feature Importances")
    plt.bar(
        range(n_top_features), importances[indices][:n_top_features], align="center"
    )
    plt.xticks(range(n_top_features), top_features, rotation=90)
    plt.xlim([-1, n_top_features])
    plt.show()

    # Create a new DataFrame with the selected top features and the target variable
    df_selected = df[top_features].copy()
    df_selected[target_column] = df[target_column]

    return df_selected
