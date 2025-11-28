# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 10:46:38 2025

@author: ACER
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def PreprocessData(train_df, test_df):
    """
    Preprocess training and test datasets:
    - Drop customerID
    - Convert TotalCharges to numeric
    - Label encode categorical features
    - Scale numeric features
    Returns: X_train, y_train, X_test, y_test, encoders, scaler
    """
    # Drop customerID
    for df in [train_df, test_df]:
        if "customerID" in df.columns:
            df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges
    for df in [train_df, test_df]:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.fillna(0, inplace=True)

    # Label Encoding
    encoders = {}
    for col in train_df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        encoders[col] = le
        if col in test_df.columns:
            test_df[col] = le.transform(test_df[col])

    # Split
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, encoders, scaler


def TuneModels(X_train, y_train):
    """
    Hyperparameter tuning for multiple classifiers using GridSearchCV
    Returns a dictionary of best models
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier()
    }

    param_grids = {
        "LogisticRegression": {"C": [0.1, 1, 10], "solver": ["lbfgs"]},
        "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "AdaBoost": {"n_estimators": [50, 100, 150], "learning_rate": [0.5, 1, 1.5]},
        "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
        "DecisionTree": {"max_depth": [None, 10, 20], "criterion": ["gini", "entropy"]},
        "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
    }

    BestModels = {}
    for name in models:
        grid = GridSearchCV(models[name], param_grids[name], cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        BestModels[name] = grid.best_estimator_
        print(f"{name} best params: {grid.best_params_} | best score: {grid.best_score_:.3f}")

    return BestModels
