import mlflow

mlflow.set_experiment("MLflow Quickstart")
mlflow.sklearn.autolog()

## ml training #########


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "random_state": 42,
    "solver": "sag",
    "max_iter": 200,
}

model = LogisticRegression(**params)
model.fit(X_train, y_train)



