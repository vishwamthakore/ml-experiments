from data.iris_v1 import IrisV1

DATA_REGISTRY = {
    "iris_v1" : IrisV1
}

def load_dataset(dataset_name):
    return DATA_REGISTRY[dataset_name]()