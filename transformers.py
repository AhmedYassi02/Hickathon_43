
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class Transformer(ABC, BaseEstimator, TransformerMixin):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame):
        pass

# EXAMPLE OF TRANSFORMER FOR CLEANING / PROCESSING


class NewTransformer(Transformer):
    def __init__(self):
        # TODO
        pass

    def fit(self, X, y=None):

        # TODO

        return self

    def transform(self, X):

        # TODO

        return X


class DateTransformer(Transformer):
    def __init__(self):
        self.date_cols = []

    def fit(self, X, y=None):
        self.date_cols = [col for col in X.columns if 'date' in col]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.date_cols:
            if col == 'meteo_date':
                X[col] = pd.to_datetime(X[col], errors='coerce').apply(
                    lambda x: np.cos(x * 2 * np.pi / 365.25))
            else:
                X.drop(col, axis=1, inplace=True)
            X.rename(columns={'meteo_date': 'date'}, inplace=True)
        return X


class DropCols(Transformer):
    def __init__(self, columns: list[str]):
        self.columns = columns
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.drop(columns=self.columns)

        print(f">> (INFO - DropCols) columns {self.columns} is/are droped.")

        return X
