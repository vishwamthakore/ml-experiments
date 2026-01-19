import pandas as pd

def get_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[['sepal length (cm)', 'sepal width (cm)']]