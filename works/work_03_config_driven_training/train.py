"""
Docstring for project.train

Get data from a config string
Get features that are obtained by processing data from a string
Get model from a string
Get params for that model
"""

import argparse
from data import data_registry
from data.base import BaseDataset
from features import feature_registry
from models import model_registry
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
from utils import load_config, flatten_dict


def main(config: dict):
    with mlflow.start_run():
        print(config)

        flattened_config = flatten_dict(nested_dict=config)
        mlflow.log_params(params=flattened_config)

        dataset: BaseDataset = data_registry.load_dataset(dataset_name=config["data"]["name"])
        X_raw, y = dataset.load()

        get_features = feature_registry.load_get_features(feature_version=config["features"]["version"])
        X = get_features(X_raw)

        print(X.columns)
        model = model_registry.get_model(config["model"]["name"], config["model"]["params"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        print(accuracy)
        mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True, help="Name of config file")
        args = parser.parse_args()
        config = load_config(config=args.config)
        main(config=config)
    except Exception as e:
        raise RuntimeError(f"Training failed with error : {e}")

