
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


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


class AltitudeTrans(Transformer):
    def __init__(self, columns):
        self.columns = columns
        pass

    def fit(self, X, y=None):

        self.max_altitude: pd.Series = X[self.columns].max()
        self.most_frequent: pd.Series = X[self.columns][
            (X[self.columns] >= 0) &
            (X[self.columns] <= self.max_altitude)
        ].mode()

        print(f"most_frequent Altitude: {self.most_frequent}")
        print(f"max altitude: {self.max_altitude}")

        return self

    
    def transform(self, X):
        
        for col in self.columns:
            X[col] = X[col].clip(upper=self.max_altitude[col]) # For high value, we cap to the max value of train
            X[col][X[col] < 0] = self.most_frequent[col] # Value < 0, we put the most frequent

        return X
    
class PartialStandardScaler(Transformer):
    """partial because only some columns can be selected for standardiation."""    

    def __init__(
            self,
            columns: list[str],
            *,
            copy: bool = True,
            with_mean: bool = True,
            with_std: bool = True
        ):
        self.columns = columns
        self.standardizer = StandardScaler(
            copy=copy,
            with_mean=with_mean,
            with_std=with_std,
        )

    def fit(self, X, y=None):

        self.standardizer.fit(X[self.columns])

        return self

    
    def transform(self, X):
        
        X_standardized_np = self.standardizer.transform(X[self.columns])

        X_standardized = pd.DataFrame(X_standardized_np, columns=self.standardizer.get_feature_names_out())

        X = pd.concat([X.drop(self.columns, axis=1), X_standardized], axis=1)

        print(f">> (INFO - PartialStandardScaler) columns {self.columns} have bean standardized")


        return X