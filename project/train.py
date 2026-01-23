"""
Docstring for project.train

Get data from a config string
Get features that are obtained by processing data from a string
Get model from a string
Get params for that model
"""

import yaml
import argparse
from pathlib import Path
from data import data_registry
from data.base import BaseDataset
from features import feature_registry
from models import model_registry
from sklearn.metrics import accuracy_score
import mlflow

def main(config: dict):
    with mlflow.start_run():
        print(config)
        mlflow.log_params(params=config)

        dataset: BaseDataset = data_registry.load_dataset(dataset_name=config["data"]["name"])
        X_raw, y = dataset.load()

        get_features = feature_registry.load_get_features(feature_version=config["features"]["version"])
        X = get_features(X_raw)

        print(X.columns)
        model = model_registry.get_model(config["model"]["name"], config["model"]["params"])
        model.fit(X, y)
        y_pred = model.predict(X)

        accuracy = accuracy_score(y_true=y, y_pred=y_pred)
        print(accuracy)
        mlflow.log_metric("accuracy", accuracy)



def load_config(config: str) -> dict:
    filepath = Path(__file__).parent / "configs" / f"{config}.yaml"

    with open(filepath) as f:
        config_dict = yaml.safe_load(f)
        print(f"Config file {config} loaded")

    return config_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Name of config file")
    args = parser.parse_args()

    config = load_config(config=args.config)
    main(config=config)
