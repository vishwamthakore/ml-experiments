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
from replay import replay
from utils import load_config, flatten_dict, get_hash, get_source_code, get_git_sha


def main(config: dict):
    mlflow.set_experiment(config["experiment"]["name"])
    
    with mlflow.start_run():
        print(config)

        flattened_config = flatten_dict(nested_dict=config, parent_key="config")
        mlflow.log_params(params=flattened_config)

        dataset: BaseDataset = data_registry.load_dataset(dataset_name=config["data"]["name"])
        X_raw, y = dataset.load()

        mlflow.log_param("data.metadata", dataset.get_metadata(X=X_raw, y=y))
        mlflow.log_param("data.fingerprint", dataset.fingerprint(X=X_raw, y=y))

        get_features = feature_registry.load_get_features(feature_version=config["features"]["version"])
        X = get_features(X_raw)

        feature_code = get_source_code(get_features)
        feature_code_hash = get_hash(feature_code)
        mlflow.log_param("feature.code_hash", feature_code_hash)
        mlflow.log_param("feature.code", feature_code)

        mlflow.log_param("git_sha", get_git_sha())

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
        
        if config["experiment"]["mode"] == "replay":
            replay(config=config)
        else:
            main(config=config)

    except Exception as e:
        raise RuntimeError(f"Training failed with error : {e}")

