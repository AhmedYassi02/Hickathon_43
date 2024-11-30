
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


##### --------------- class yael --------------------------###
class CleanYael(Transformer):
    ## prépare les features  "insee_%_agri" et "meteo_rain_height"
    def __init__(self):
        # Initialize placeholders for the medians
        self.insee_median = None
        self.meteo_median = None

    def fit(self, X, y=None):
        # Column names to clean
        insee = "insee_%_agri"
        meteo = "meteo_rain_height"

        # Standardize the `insee_%_agri` column
        X[insee] = pd.to_numeric(X[insee], errors='coerce')  # Converts strings to NaN
        X[insee] = X[insee].astype(float)  # Ensure column is float
        print(f">> (Info) Column {insee} has been standardized to numeric.")

        # Compute and store the medians after standardizing
        self.insee_median = X[insee].median()
        self.meteo_median = X[meteo].median()
        return self

    def transform(self, X):
        # Column names
        insee = "insee_%_agri"
        meteo = "meteo_rain_height"

        # Ensure the `insee_%_agri` column is standardized (in case it wasn't during fit)
        X[insee] = pd.to_numeric(X[insee], errors='coerce')
        X[insee] = X[insee].astype(float)

        # Fill missing values with the computed medians
        X[insee] = X[insee].fillna(self.insee_median)
        X[meteo] = X[meteo].fillna(self.meteo_median)

        print(f">> (Info) Missing values in {insee} filled with median: {self.insee_median}")
        print(f">> (Info) Missing values in {meteo} filled with median: {self.meteo_median}")

        return X