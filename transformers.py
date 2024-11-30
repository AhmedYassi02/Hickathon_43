
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Union


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
    '''
    NEEDS : meteo_date
    INPUT : / 
    RETURNS : meteo_date (processed)
    DROPS : All other dates
    '''

    def __init__(self):
        self.date_cols = []
        self.time_cols = []

    def fit(self, X, y=None):
        self.date_cols = [col for col in X.columns if 'date' in col]
        self.time_cols = [col for col in X.columns if 'time' in col]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.date_cols:
            if col == 'meteo_date':
                X[col] = pd.to_datetime(X[col], errors='coerce').dt.dayofyear.apply(
                    lambda x: np.cos((x - 1) * 2 * np.pi / 365.25))
            else:
                X.drop(col, axis=1, inplace=True)
            #X.rename(columns={'meteo_date': 'date'}, inplace=True)

        for col in self.time_cols:
            X[col] = X[col].apply(lambda x: np.cos(x * 2 * np.pi / 24))
        return X


class DropCols(Transformer):
    def __init__(self, columns: list[str]):
        self.columns = columns
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.drop(columns=self.columns,errors='ignore') #on ingore les erreurs

        print(f">> (INFO - DropCols) columns {self.columns} is/are droped.")

        return X


class AltitudeTrans(Transformer):
    '''
    NEEDS : ["piezo_station_altitude", "meteo_altitude"]
    INPUT : ["piezo_station_altitude", "meteo_altitude"]
    RETURNS : ["piezo_station_altitude", "meteo_altitude"]
    DROPS : None
    
    '''
    

    def __init__(self, columns):
        self.columns = columns
        pass

    def fit(self, X, y=None):

        self.max_altitude: pd.Series = X[self.columns].max()
        self.most_frequent: pd.Series = X[self.columns][
            (X[self.columns] >= 0) &
            (X[self.columns] <= self.max_altitude)
        ].mode()
        self.mean = X[self.columns].mean()

        return self

    def transform(self, X):

        for col in self.columns:
            # For high value, we cap to the max value of train
            X[col] = X[col].clip(upper=self.max_altitude[col])
            # Value < 0, we put the most frequent
            X.loc[X[col] < 0, col] = self.most_frequent[col]

            X = X.fillna(self.mean[col])

        return X


class PartialStandardScaler(Transformer):
    '''partial because only some columns can be selected for standardiation

    #NEEDS : /
    # INPUT : numeric_cols 
    # RETURNS : standardized numeric columns 
    # DROPS : None
    '''
       
    def __init__(
        self,
        columns:  Union[list[str], str],
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

        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):

        if self.columns == "all":
            self.columns = X.columns.to_list()

        assert X.apply(lambda x: pd.api.types.is_numeric_dtype(
            x)).all(), "Some columns to standardize are not numerics"

        self.standardizer.fit(X[self.columns])

        return self

    def transform(self, X):

        assert X.apply(lambda x: pd.api.types.is_numeric_dtype(
            x)).all(), "Some columns to standardize are not numerics"

        X_standardized_np = self.standardizer.transform(X[self.columns])

        X_standardized = pd.DataFrame(
            X_standardized_np, columns=self.standardizer.get_feature_names_out(), index=X.index)

        X = pd.concat([X.drop(self.columns, axis=1), X_standardized], axis=1)

        print(
            f">> (INFO - PartialStandardScaler) columns {self.columns} have bean standardized")

        return X

##### --------------- class yael --------------------------###


class CleanFeatures(Transformer):
    ''' prépare les features  "insee_%_agri" et "meteo_rain_height"

    NEEDS : ["insee_%_agri","meteo_rain_height"]
    INPUT : ["insee_%_agri","meteo_rain_height"]
    RETURNS : ["insee_%_agri","meteo_rain_height"] (cleaned)
    DROPS : None

    '''

    def __init__(self, cols):
        # Initialize placeholders for the medians
        self.insee_median = None
        self.meteo_median = None
        self.cols = cols

        if "insee_%_agri" in self.cols:
            self.handle_insee = True
        else:
            self.handle_insee = False
        if "meteo_rain_height" in self.cols:
            self.handle_meteo = True
        else:
            self.handle_meteo = False

    def fit(self, X, y=None):
        # Column names to clean
        insee = "insee_%_agri"
        meteo = "meteo_rain_height"

        # Standardize the `insee_%_agri` column
        if self.handle_insee:

            # Converts strings to NaN
            X[insee] = pd.to_numeric(X[insee], errors='coerce')
            X[insee] = X[insee].astype(float)  # Ensure column is float
            print(
                f">> (Info) Column {insee} has been standardized to numeric.")
            self.insee_median = X[insee].median()

        # Compute and store the medians after standardizing
        if self.handle_meteo:
            self.meteo_median = X[meteo].median()

        return self

    def transform(self, X):
        # Column names
        insee = "insee_%_agri"
        meteo = "meteo_rain_height"

        if self.handle_insee:

            # Ensure the `insee_%_agri` column is standardized (in case it wasn't during fit)
            X[insee] = pd.to_numeric(X[insee], errors='coerce')
            X[insee] = X[insee].astype(float)

        # Fill missing values with the computed medians
            X[insee] = X[insee].fillna(self.insee_median)

            print(
                f">> (Info) Missing values in {insee} filled with median: {self.insee_median}")

        if self.handle_meteo:
            X[meteo] = X[meteo].fillna(self.meteo_median)
            print(
                f">> (Info) Missing values in {meteo} filled with median: {self.meteo_median}")

        return X


class TemperaturePressionTrans(Transformer):

    '''
    NEEDS : ['piezo_station_department_code', 'piezo_measurement_date']
    INPUT : ['meteo_amplitude_tn_tx','meteo_temperature_avg','meteo_temperature_avg_threshold','meteo_temperature_min','meteo_temperature_min_50cm','meteo_temperature_min_ground','meteo_temperature_avg_tntm','meteo__pressure_saturation_avg','meteo_temperature_max']
    Input reduit : ['meteo_temperature_avg','meteo_temperature_min','meteo__pressure_saturation_avg','meteo_temperature_max']
    RETURNS : les colonnes de l'input, avec valeurs manquantes completées, et dropped la ou ya plus de 60% valeur manquantes
    '''


    def __init__(self, columns: list[str]):
        self.columns = columns
        pass

    def fit(self, X, y=None):
        return self

    
    def transform(self, X):
        #Partie 1 : supprimé les colonnes avec + de 60% de valeurs manquantes
        
        # Select only the specified columns
        relevant_cols = [col for col in self.columns if col in X.columns]

        # Calculate the threshold for missing values
        threshold = 0.6 * len(X)

        # Identify columns to drop within the relevant columns
        cols_to_drop = [col for col in relevant_cols if X[col].isna().sum() > threshold]

        # Drop the identified columns
        X = X.drop(columns=cols_to_drop)

        #Traitement des valeurs manquantes : moyenne sur le département à la meme date ou meme date si données manquantes
        
        for column in self.columns:
            if column in X.columns:
                # Check if the column contains NaN values
                if X[column].isna().sum() > 0:
                    # Fill NaN by department and date mean
                    moyennes_departement_date = X.groupby(['piezo_station_department_code', 'piezo_measurement_date'])[column].transform('mean')
                    X[column] = X[column].fillna(moyennes_departement_date)

                    # Step 3: Fill any remaining NaN by the mean of the date (ignoring the department)
                    moyennes_date = X.groupby('piezo_measurement_date')[column].transform('mean')
                    X[column] = X[column].fillna(moyennes_date)

        return X


class CleanLatLon(Transformer):
    """
    Nettoyage des données relatives aux coordonnées géographiques
    - Inversion lat/lon pour les stations météos
    - Application d'un threshold (float -> boolean) pour la distance

     NEEDS: ["distance_piezo_meteo",'piezo_station_longitude','piezo_station_latitude','meteo_latitude','meteo_longitude']
    INPUT: /
    RETURNS : 
    DROPS: A lot (cf en bas du code)

    """

    def __init__(self, apply_threshold=True, dist_to_meteo_threshold=None):
        self.apply_threshold = apply_threshold
        self.threshold = dist_to_meteo_threshold

    def fit(self, X, y=None):
        if self.apply_threshold and self.threshold is None:
            self.threshold = X["distance_piezo_meteo"].quantile(0.95)
        return self

    def transform(self, X):
        X = X.copy()

        temp = X["meteo_longitude"].copy()
        X["meteo_longitude"] = X["meteo_latitude"].copy()
        X["meteo_latitude"] = temp

        if self.apply_threshold:
            X["near_meteo"] = (X["distance_piezo_meteo"] <= self.threshold).astype(float)
            X["distance_piezo_meteo"] = X["near_meteo"]

        drop_cols = [
            "meteo_longitude",
            "meteo_latitude",
            "hydro_longitude",
            "hydro_latitude",
            "prelev_longitude_0",
            "prelev_latitude_0",
            "prelev_longitude_1",
            "prelev_latitude_1",
            "prelev_longitude_2",
            "prelev_latitude_2",
            "near_meteo"
        ]
        X.drop(columns=drop_cols, inplace=True,errors='ignore') #errors=ignore pour qu'il n y ait pas d'erreurs is la colonne n'existe pas 

        return X


class MissingCat(Transformer):
    """Créer une categorie 'missing' pour les valeurs manquantes car dans le data test il ya bcp de valeur manquante dans ces colonnes catégorielles
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        print(
            f">> (INFO) missing categorie is added to columns {self.columns}")
        return self

    def transform(self, X):
        X = (X.copy()
             .fillna('missing', axis=1)
             )
        return X


class DummyTransformer(Transformer):
    """Transoformer les categories en valeurs entieres pour les colonnes catégorielles
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        print(f">> (INFO) columns {self.columns} are transformed to dummies")
        return self

    def transform(self, X):
        X = pd.get_dummies(X, columns=self.columns)
        return X
