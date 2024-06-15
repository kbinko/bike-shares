##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 12-01-2023
# Added the regression selection by Konrad Binko on 14.06.2024

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    mean_squared_error,
)
import pandas as pd
import numpy as np
import copy


class ClassificationAlgorithms:

    # Forward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def forward_selection(self, max_features, X_train, y_train):
        # Start with no features.
        ordered_features = []
        ordered_scores = []
        selected_features = []
        ca = ClassificationAlgorithms()
        prev_best_perf = 0

        # Select the appropriate number of features.
        for i in range(0, max_features):
            print(i)

            # Determine the features left to select.
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = 0
            best_attribute = ""

            # For all features we can still select...
            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

                # Determine the accuracy of a decision tree learner if we were to add
                # the feature.
                (
                    pred_y_train,
                    pred_y_test,
                    prob_training_y,
                    prob_test_y,
                ) = ca.decision_tree(
                    X_train[temp_selected_features],
                    y_train,
                    X_train[temp_selected_features],
                )
                perf = accuracy_score(y_train, pred_y_train)

                # If the performance is better than what we have seen so far (we aim for high accuracy)
                # we set the current feature to the best feature and the same for the best performance.
                if perf > best_perf:
                    best_perf = perf
                    best_feature = f
            # We select the feature with the best performance.
            selected_features.append(best_feature)
            prev_best_perf = best_perf
            ordered_features.append(best_feature)
            ordered_scores.append(best_perf)
        return selected_features, ordered_features, ordered_scores

    # Apply a neural network for classification upon the training data (with the specified composition of
    # hidden layers and number of iterations), and use the created network to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def feedforward_neural_network(
        self,
        train_X,
        train_y,
        test_X,
        hidden_layer_sizes=(100,),
        max_iter=2000,
        activation="logistic",
        alpha=0.0001,
        learning_rate="adaptive",
        gridsearch=True,
        print_model_details=False,
    ):

        if gridsearch:
            tuned_parameters = [
                {
                    "hidden_layer_sizes": [
                        (5,),
                        (10,),
                        (25,),
                        (100,),
                        (
                            100,
                            5,
                        ),
                        (
                            100,
                            10,
                        ),
                    ],
                    "activation": [activation],
                    "learning_rate": [learning_rate],
                    "max_iter": [1000, 2000],
                    "alpha": [alpha],
                }
            ]
            nn = GridSearchCV(
                MLPClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            # Create the model
            nn = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                max_iter=max_iter,
                learning_rate=learning_rate,
                alpha=alpha,
            )

        # Fit the model
        nn.fit(
            train_X,
            train_y.values.ravel(),
        )

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        pred_prob_training_y = nn.predict_proba(train_X)
        pred_prob_test_y = nn.predict_proba(test_X)
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def support_vector_machine_with_kernel(
        self,
        train_X,
        train_y,
        test_X,
        kernel="rbf",
        C=1,
        gamma=1e-3,
        gridsearch=True,
        print_model_details=False,
    ):
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {"kernel": ["rbf", "poly"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]}
            ]
            svm = GridSearchCV(
                SVC(probability=True), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            svm = SVC(
                C=C, kernel=kernel, gamma=gamma, probability=True, cache_size=7000
            )

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        pred_prob_training_y = svm.predict_proba(train_X)
        pred_prob_test_y = svm.predict_proba(test_X)
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def support_vector_machine_without_kernel(
        self,
        train_X,
        train_y,
        test_X,
        C=1,
        tol=1e-3,
        max_iter=1000,
        gridsearch=True,
        print_model_details=False,
    ):
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {"max_iter": [1000, 2000], "tol": [1e-3, 1e-4], "C": [1, 10, 100]}
            ]
            svm = GridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring="accuracy")
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model

        distance_training_platt = 1 / (1 + np.exp(svm.decision_function(train_X)))
        pred_prob_training_y = (
            distance_training_platt / distance_training_platt.sum(axis=1)[:, None]
        )
        distance_test_platt = 1 / (1 + np.exp(svm.decision_function(test_X)))
        pred_prob_test_y = (
            distance_test_platt / distance_test_platt.sum(axis=1)[:, None]
        )
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a nearest neighbor approach for classification upon the training data (with the specified value for
    # k), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def k_nearest_neighbor(
        self,
        train_X,
        train_y,
        test_X,
        n_neighbors=5,
        gridsearch=True,
        print_model_details=False,
    ):
        # Create the model
        if gridsearch:
            tuned_parameters = [{"n_neighbors": [1, 2, 5, 10]}]
            knn = GridSearchCV(
                KNeighborsClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_prob_training_y = knn.predict_proba(train_X)
        pred_prob_test_y = knn.predict_proba(test_X)
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a decision tree approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, and the export path and files if print_model_details=True)
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def decision_tree(
        self,
        train_X,
        train_y,
        test_X,
        min_samples_leaf=50,
        criterion="gini",
        print_model_details=False,
        export_tree_path="Example_graphs/Chapter7/",
        export_tree_name="tree.dot",
        gridsearch=True,
    ):
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "criterion": ["gini", "entropy"],
                }
            ]
            dtree = GridSearchCV(
                DecisionTreeClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            dtree = DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf, criterion=criterion
            )

        # Fit the model

        dtree.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_prob_training_y = dtree.predict_proba(train_X)
        pred_prob_test_y = dtree.predict_proba(test_X)
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)
        frame_prob_training_y = pd.DataFrame(
            pred_prob_training_y, columns=dtree.classes_
        )
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(dtree.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            print("Feature importance decision tree:")
            for i in range(0, len(dtree.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(dtree.feature_importances_[ordered_indices[i]])
            tree.export_graphviz(
                dtree,
                out_file=export_tree_path + export_tree_name,
                feature_names=train_X.columns,
                class_names=dtree.classes_,
            )

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a naive bayes approach for classification upon the training data
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def naive_bayes(self, train_X, train_y, test_X):
        # Create the model
        nb = GaussianNB()

        # Fit the model
        nb.fit(train_X, train_y)

        # Apply the model
        pred_prob_training_y = nb.predict_proba(train_X)
        pred_prob_test_y = nb.predict_proba(test_X)
        pred_training_y = nb.predict(train_X)
        pred_test_y = nb.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a random forest approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, the number of trees, and if we should print some of the details of the
    # model print_model_details=True) and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def random_forest(
        self,
        train_X,
        train_y,
        test_X,
        n_estimators=10,
        min_samples_leaf=5,
        criterion="gini",
        print_model_details=False,
        gridsearch=True,
    ):

        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "n_estimators": [10, 50, 100],
                    "criterion": ["gini", "entropy"],
                }
            ]
            rf = GridSearchCV(
                RandomForestClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
            )

        # Fit the model

        rf.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        pred_prob_training_y = rf.predict_proba(train_X)
        pred_prob_test_y = rf.predict_proba(test_X)
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(rf.feature_importances_), key=lambda x: x[1], reverse=True
                )
            ]
            print("Feature importance random forest:")
            for i in range(0, len(rf.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(rf.feature_importances_[ordered_indices[i]])

        return (
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
        )


# Function to evaluate different regression models
def evaluate_regression_models(
    df,
    target_column,
    models_to_evaluate=None,
    test_size=0.2,
    random_state=42,
):
    # Default models to evaluate if none are provided
    if models_to_evaluate is None:
        models_to_evaluate = [
            "RandomForest",
            "NeuralNetwork",
            "SVR",
            "LinearSVR",
            "KNN",
            "DecisionTree",
        ]

    # Split the data into features and target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Define the models to evaluate
    models = {
        "RandomForest": RandomForestRegressor(random_state=random_state),
        "NeuralNetwork": MLPRegressor(random_state=random_state),
        "SVR": SVR(),
        "LinearSVR": LinearSVR(random_state=random_state),
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
    }

    # Grid search parameters for each model
    param_grids = {
        "RandomForest": {"n_estimators": [100, 200], "min_samples_leaf": [1, 2, 4]},
        "NeuralNetwork": {
            "hidden_layer_sizes": [(50,), (100,)],
            "max_iter": [1000, 2000],
        },
        "SVR": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        "LinearSVR": {"C": [0.1, 1, 10], "max_iter": [1000, 2000]},
        "KNN": {"n_neighbors": [3, 5, 7]},
        "DecisionTree": {
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["absolute_error", "poisson", "squared_error", "friedman_mse"],
            "splitter": ["best", "random"],
            "max_depth": [None, 10, 20],
        },
    }

    # DataFrame to store the results
    results = pd.DataFrame(columns=["Model", "MAE", "MSE", "R^2"])

    # Variable to store the best model and its performance
    best_model = None
    best_mae = float("inf")

    # Evaluate each model
    for model_name in models_to_evaluate:
        model = models[model_name]
        print(f"Evaluating {model_name}...")

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            model, param_grids[model_name], cv=5, scoring="neg_mean_absolute_error"
        )
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_model_for_current = grid_search.best_estimator_

        # Predict on validation set
        y_pred = best_model_for_current.predict(X_val)

        # Calculate performance metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Print the performance of the current model
        print(f"{model_name} Performance:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R^2 Score: {r2}\n")

        # Append the results to the DataFrame
        result_row = pd.DataFrame(
            {"Model": [model_name], "MAE": [mae], "MSE": [mse], "R^2": [r2]}
        )
        results = pd.concat([results, result_row], ignore_index=True)
        # Check if this model is the best so far
        if mae < best_mae:
            best_mae = mae
            best_model = best_model_for_current

    return results, best_model


def evaluate_feature_sets(
    df, target_column, feature_sets, test_size=0.2, random_state=42
):
    results = pd.DataFrame(columns=["Feature Set", "MAE", "MSE", "R^2"])
    best_mae = float("inf")
    best_feature_set = None
    best_model = None

    # Define the Neural Network model
    nn_model = MLPRegressor(random_state=random_state)

    # Grid search parameters for Neural Network
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,)],
        "max_iter": [1000, 2000],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001],
    }

    for i, feature_set in enumerate(feature_sets):
        print(f"Evaluating feature set {i + 1}: {feature_set}")

        # Split the data into features and target variable
        X = df[feature_set]
        y = df[target_column]

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            nn_model, param_grid, cv=5, scoring="neg_mean_absolute_error"
        )
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_nn_model = grid_search.best_estimator_

        # Predict on validation set
        y_pred = best_nn_model.predict(X_val)

        # Calculate performance metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Print the performance of the current feature set
        print(f"Feature Set {i + 1} Performance:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R^2 Score: {r2}\n")

        # Append the results to the DataFrame
        results = results.append(
            {"Feature Set": i + 1, "MAE": mae, "MSE": mse, "R^2": r2}, ignore_index=True
        )

        # Check if this feature set is the best so far
        if mae < best_mae:
            best_mae = mae
            best_feature_set = feature_set
            best_model = best_nn_model

    return results, best_feature_set, best_model
