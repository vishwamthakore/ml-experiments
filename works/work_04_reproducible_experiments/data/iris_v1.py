from data.base import BaseDataset
from sklearn import datasets

class IrisV1(BaseDataset):
    def load(self):
        X, y = datasets.load_iris(return_X_y=True, as_frame=True)
        return X, y
