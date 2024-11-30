
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.model_selection import train_test_split


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


class DropNaRate(Transformer):
    def __init__(self, rate: float):
        self.rate = rate

    def fit(self, X, y=None):

        perc_na = X.isna().sum()/X.shape[0]
        self.cols_to_drop: pd.Series = perc_na[perc_na > self.rate].index

        print(f">> (Info) Droped columns : {self.cols_to_drop.to_list()}")

        return self

    def transform(self, X):

        X = X.drop(columns=self.cols_to_drop)

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

        return self

    def transform(self, X):

        for col in self.columns:
            # For high value, we cap to the max value of train
            X[col] = X[col].clip(upper=self.max_altitude[col])
            # Value < 0, we put the most frequent
            X.loc[X[col] < 0, col] = self.most_frequent[col]

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

        X_standardized = pd.DataFrame(
            X_standardized_np, columns=self.standardizer.get_feature_names_out())

        X = pd.concat([X.drop(self.columns, axis=1), X_standardized], axis=1)

        print(f">> (INFO - PartialStandardScaler) columns {self.columns} have bean standardized")
    
        return X

##### --------------- class yael --------------------------###


class CleanYael(Transformer):
    # prépare les features  "insee_%_agri" et "meteo_rain_height"
    def __init__(self):
        # Initialize placeholders for the medians
        self.insee_median = None
        self.meteo_median = None

    def fit(self, X, y=None):
        # Column names to clean
        insee = "insee_%_agri"
        meteo = "meteo_rain_height"

        # Standardize the `insee_%_agri` column
        # Converts strings to NaN
        X[insee] = pd.to_numeric(X[insee], errors='coerce')
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

        print(
            f">> (Info) Missing values in {insee} filled with median: {self.insee_median}")
        print(
            f">> (Info) Missing values in {meteo} filled with median: {self.meteo_median}")

        return X


if __name__ == "__main__":

    path_src_dataset = Path("./data/src/X_train_Hi5.csv")

    out_folder_dataset = Path("./data/cleaned")
    # Create the folder if it doesn't exist
    out_folder_dataset.mkdir(parents=True, exist_ok=True)

    out_folder_config = Path("./data/cleaned/pipelines")
    out_folder_config.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path_src_dataset)

    target = "piezo_groundwater_level_category"

    X = df.drop(columns=target)

    mapping = {'Very Low': 0, 'Low': 1, 'Average': 2, 'High': 3, 'Very High': 4}
    y = df[target].map(mapping)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    # Apply the transformers selected
    pipeline = Pipeline(steps=[
        ("DropNaRate", DropNaRate(0.7)),
        ("CleanYael", CleanYael()),
        ("Altitude", AltitudeTrans(columns=["piezo_station_altitude", "meteo_altitude"])),
        # ... Add others transformations
    ])


    print("Pipelin ongoing...")
    processed_X_train = pipeline.fit_transform(X_train)
    processed_X_val = pipeline.transform(X_val)
