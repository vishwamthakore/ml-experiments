import yaml
from pathlib import Path
import hashlib
import inspect
import git

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

def get_source_code(func) -> str:
    return inspect.getsource(func)

def get_hash(input_string: str) -> str:
    return hashlib.md5(input_string.encode()).hexdigest()

def get_git_sha() -> str:
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha