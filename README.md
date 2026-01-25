# ML Experiments – Config-Driven Training (Learning Project)

## What is this project

This repository is a learning project to understand how ML experimentation,
reproducibility, and config-driven training systems are designed in production
ML platforms.

The goal is **not** to build a new MLFlow replacement, but to understand
*how components are wired together*:
data → features → model → experiment tracking.

This is to understand how a framework can be built on top of mlflow. 
It will be opinionated with guidelines, one correct way to do things so it can scale.

## Why this project exists

In real ML systems:
- experiments must be reproducible
- model training must be configurable (not hardcoded)
- changes to data, features, or models must be traceable

This project explores these ideas by building a **minimal but realistic**
training pipeline using:
- Python scripts (not notebooks)
- YAML configs
- registries for data / features / models
- MLflow for experiment tracking


## What this project currently supports

- Config-driven training via YAML
- Dataset versioning via registry (e.g. `iris_v1`)
- Feature versioning via functions (e.g. `iris_v1_features_v1`)
- Model selection via registry
- Parameter logging to MLflow
- Reproducible train/test split


## How to run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Start MLflow UI (optional)

```bash
mlflow server --port 5000
```


3. Run training

```bash
python train.py --config exp_v1
```

This will:
- load the dataset specified in the config
- apply the selected feature version
- train the selected model with given parameters
- log parameters and metrics to MLflow


## Project structure

## Project Structure

```text
.
├── train.py                 # Entry point for training
├── configs/
│   └── exp_v1.yaml           # Experiment configuration
├── data/
│   ├── base.py               # Dataset interface
│   ├── data_registry.py      # Dataset registry
│   └── iris_v1.py            # Iris dataset (v1)
├── features/
│   ├── feature_registry.py   # Feature registry
│   └── iris_v1_features_v1.py
├── models/
│   └── model_registry.py     # Model registry
├── utils.py                 # Shared utilities
└── README.md
```

## Design principles

- Training logic lives in scripts, not notebooks
- Configuration drives behavior, not code edits
- Each experiment run should be reproducible
- Errors are handled at the entry point (not scattered)

## What this project does NOT do (yet)

- No data version snapshots (only named versions)
- No feature metadata tracking
- No model registry
- No inference or serving
- No CI / automation

These will be explored incrementally as learning steps.

## Status

This is an incremental learning project.
The README and architecture evolve with each “work”.