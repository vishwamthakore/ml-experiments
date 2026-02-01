import mlflow
from data import data_registry
from features import feature_registry
from utils import get_hash, get_source_code


def get_run_by_name(run_name):
    runs_df = mlflow.search_runs(search_all_experiments=True)
    run_row = runs_df[(runs_df["tags.mlflow.runName"] == run_name)]
    run = mlflow.get_run(run_id=run_row["run_id"][0])
    return run


def replay(config: dict):
    experiment_name = config["experiment"]["name"]
    run_name = config["experiment"]["runName"]
    history_run = get_run_by_name(run_name=run_name)
    comparison_report = build_comparison_report(history_run=history_run, config=config)
    verdict = get_verdict_from_comaprison_report(comparison_report)

    if verdict:
        print("Contract satisfied. Safe to rerun")
    else:
        print("Contract violated. Rerun blocked")


def build_comparison_report(history_run, config: dict) -> dict:
    old_data_fingerprint = history_run.data.params["data.fingerprint"]
    old_feature_code_hash = history_run.data.params["feature.code_hash"]

    # new data fingerprint
    dataset = data_registry.load_dataset(dataset_name=config["data"]["name"])
    X_raw, y = dataset.load()
    new_data_fingerprint = dataset.fingerprint(X=X_raw, y=y)

    # new feature code hash
    get_features = feature_registry.load_get_features(config["features"]["version"])
    feature_code = get_source_code(get_features)
    new_feature_code_hash = get_hash(feature_code)

    comparison_report = {}

    comparison_report["data_fingerprint"] = {
        "old": old_data_fingerprint,
        "new": new_data_fingerprint,
        "match": old_data_fingerprint == new_data_fingerprint,
    }

    comparison_report["feature_code_hash"] = {
        "old": old_feature_code_hash,
        "new": new_feature_code_hash,
        "match": old_feature_code_hash == new_feature_code_hash,
    }
    return comparison_report


def get_verdict_from_comaprison_report(comparison_report: dict) -> bool:
    return all(check["match"] for check in comparison_report.values())