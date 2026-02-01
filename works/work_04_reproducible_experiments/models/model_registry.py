from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier
}

def get_model(model_name: str, params: dict):
    return MODEL_REGISTRY[model_name](**params)
