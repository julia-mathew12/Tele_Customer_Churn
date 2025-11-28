import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def GetHyperparams(model):
    """
    Returns a string with the key hyperparameters for each model type
    """
    if isinstance(model, LogisticRegression):
        return f"max_iter={model.max_iter}, C={getattr(model, 'C', 1.0)}"
    elif isinstance(model, RandomForestClassifier):
        return f"n_estimators={model.n_estimators}, max_depth={model.max_depth}"
    elif isinstance(model, SVC):
        return f"kernel={model.kernel}, C={model.C}"
    elif isinstance(model, AdaBoostClassifier):
        return f"n_estimators={model.n_estimators}, learning_rate={model.learning_rate}"
    elif isinstance(model, XGBClassifier):
        return f"n_estimators={model.n_estimators}, learning_rate={model.learning_rate}"
    elif isinstance(model, DecisionTreeClassifier):
        return f"max_depth={model.max_depth}, criterion={model.criterion}"
    elif isinstance(model, KNeighborsClassifier):
        return f"n_neighbors={model.n_neighbors}, weights={model.weights}"
    return "N/A"

def TrainAndEvaluate(X_train, y_train, X_test, y_test):
    """
    Train multiple classifiers, evaluate performance, and return metrics and confusion matrices.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="linear", probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    summary_data = {}
    conf_matrices = {}

    for name, model in models.items():
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics in decimal format
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

        # Remarks logic
        if acc >= 0.75 and precision >= 0.7 and recall >= 0.7:
            remarks = "Excellent model"
        elif precision > recall:
            remarks = "High precision, lower recall"
        elif recall > precision:
            remarks = "High recall, lower precision"
        else:
            remarks = "Balanced model"

        summary_data[name] = {
            "Classifier": model.__class__.__name__,
            "Epochs": str(model.max_iter) if hasattr(model, "max_iter") else "N/A",
            "Hyperparameters": GetHyperparams(model),
            "Accuracy": round(acc, 2),
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1 Score": round(f1, 2),
            "Remarks": remarks
        }

        conf_matrices[name] = confusion_matrix(y_test, y_pred)

    return summary_data, conf_matrices, models
