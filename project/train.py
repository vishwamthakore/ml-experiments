import mlflow
import importlib
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

FEATURE_VERSION = "v2"
MAX_DEPTH = 3
MODEL_TYPE = "DECISON_TREE"


def load_get_features(feature_version: str):
    """Returns get_features function from module"""
    module = importlib.import_module(f"features.{feature_version}")
    return module.get_features


def main():
    # Prepare data
    X_raw, y = datasets.load_iris(return_X_y=True, as_frame=True)
    get_features = load_get_features(feature_version=FEATURE_VERSION)
    X = get_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    with mlflow.start_run():
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("feature_version", FEATURE_VERSION)
        mlflow.log_param("model_type", MODEL_TYPE)

        model = DecisionTreeClassifier(max_depth=MAX_DEPTH)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=predictions)

        mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    main()
