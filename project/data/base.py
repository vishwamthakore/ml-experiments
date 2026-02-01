import hashlib
import json
import pandas as pd

class BaseDataset:
    def load(self):
        raise NotImplementedError()
    
    def get_metadata(self, X: pd.DataFrame, y: pd.Series) -> str:
        metadata = {
            "X_shape": X.shape,
            "X_columns": sorted(list(X.columns)),
            "y_shape": y.shape
        }
        return json.dumps(metadata)

    def fingerprint(self, X: pd.DataFrame, y: pd.Series) -> str:
        metadata = self.get_metadata(X, y)
        fingerprint = hashlib.md5(metadata.encode()).hexdigest()
        return fingerprint