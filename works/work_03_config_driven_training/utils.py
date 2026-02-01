import yaml
from pathlib import Path

def load_config(config: str) -> dict:
    filepath = Path(__file__).parent / "configs" / f"{config}.yaml"

    with open(filepath) as f:
        config_dict = yaml.safe_load(f)
        print(f"Config file {config} loaded")

    return config_dict

def flatten_dict(nested_dict: dict, parent_key: str = ""):
    final = {}
    for key, value in nested_dict.items():
        full_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict):
            flattened_dict = flatten_dict(nested_dict=value, parent_key=full_key)
            final.update(flattened_dict)
        else:
            final[full_key] = value
    return final