
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from typing import Union
from sklearn.preprocessing import OneHotEncoder


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

        # Ne pas drop "meteo_temperature_min_ground"
        if 'meteo_temperature_min_ground' in self.cols_to_drop:
            self.cols_to_drop = self.cols_to_drop.drop('meteo_temperature_min_ground',errors='ignore') 

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
            # X.rename(columns={'meteo_date': 'date'}, inplace=True)

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

        # on ingore les erreurs
        X = X.drop(columns=self.columns, errors='ignore')

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

            X[col] = X[col].fillna(self.mean[col])

        return X


class Prelev(Transformer):

    def __init__(
        self,
        columns: list[str],
        usage_label_max_categories: int,
        mode_label_max_categories: int,
        scale: int,  # in [1, 2, 3]
    ):
        self.columns = columns
        self.scale = scale
        self.mode_label_max_categories = mode_label_max_categories
        self.usage_label_max_categories = usage_label_max_categories

        self.usage_oh_encoders: list[OneHotEncoder] = [
            OneHotEncoder(
                max_categories=usage_label_max_categories,
            )
            for i in range(self.scale)
        ]

        self.mode_oh_encoders: list[OneHotEncoder] = [
            OneHotEncoder(
                max_categories=mode_label_max_categories,
            )
            for i in range(self.scale)
        ]

    def fit(self, X, y=None):

        for i in range(self.scale):
            self.usage_oh_encoders[i].fit(
                pd.DataFrame(X[f"prelev_usage_label_{i}"]))
            self.mode_oh_encoders[i].fit(pd.DataFrame(
                X[f"prelev_volume_obtention_mode_label_{i}"]))

        # self.mean = X[self.columns].mean(numeric_only=True)

        return self

    def transform(self, X):

        for i in range(self.scale):
            # X[f"prelev_volume_{i}"] = X[f"prelev_volume_{i}"].fillna(
            #     self.mean[f"prelev_volume_{i}"])
            X_usage = self.usage_oh_encoders[i].transform(
                pd.DataFrame(X[f"prelev_usage_label_{i}"])).toarray()
            X_mode = self.mode_oh_encoders[i].transform(pd.DataFrame(
                X[f"prelev_volume_obtention_mode_label_{i}"])).toarray()

            X_usage_df = pd.DataFrame(
                X_usage, columns=self.usage_oh_encoders[i].get_feature_names_out(), index=X.index)
            X_mode_df = pd.DataFrame(
                X_mode, columns=self.mode_oh_encoders[i].get_feature_names_out(), index=X.index)

            X = pd.concat([
                X.drop(
                    columns=[f"prelev_usage_label_{i}", f"prelev_volume_obtention_mode_label_{i}"]),
                X_usage_df,
                X_mode_df
            ], axis=1)

        for i in range(self.scale, 3):
            X = X.drop(columns=[
                       f"prelev_volume_{i}", f"prelev_usage_label_{i}", f"prelev_volume_obtention_mode_label_{i}"])

        for i in range(self.scale):
            # mean = self.mean[f"prelev_volume_{i}"]
            # print(
            #     f">> (Info - Prelev) 'prelev_volume_{i}' has been filledna with mean = {mean}")
            print(
                f">> (Info - Prelev) 'prelev_usage_label_{i}' has been one-hot-encoded in {len(self.usage_oh_encoders[i].get_feature_names_out())} features")
            print(
                f">> (Info - Prelev) 'prelev_volume_obtention_mode_label_{i}' has been one-hot-encoded in {len(self.mode_oh_encoders[i].get_feature_names_out())} features")

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

    NEEDS : ["piezo_station_department_code", "meteo_date"]
    INPUT : ['insee_%_agri', 'meteo_rain_height', 'insee_pop_commune', 'insee_med_living_level', 'insee_%_ind', 'insee_%_const']
    RETURNS : ['insee_%_agri', 'meteo_rain_height', 'insee_pop_commune', 'insee_med_living_level', 'insee_%_ind', 'insee_%_const']] (cleaned)
    DROPS : None

    Exemple d'appel :
    cols = ['insee_%_agri', 'meteo_rain_height', 'insee_pop_commune', 'insee_med_living_level', 'insee_%_ind', 'insee_%_const']
    cleaner = CleanFeatures(cols)

    '''

    def __init__(self, cols_to_handle, department_col="piezo_station_department_code", date_col="meteo_date"):
        self.department_col = department_col
        self.date_col = date_col
        self.cols_to_handle = cols_to_handle
        self.department_medians = {}
        self.global_means = {}
        self.rain_means_by_month_department = None

    def fit(self, X, y=None):
        print(f">> (Info) Calculating medians and means for {self.cols_to_handle}")

        # Ensure all columns in cols_to_handle are numeric
        for col in self.cols_to_handle:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Calculate medians for columns grouped by department and global means
        for col in self.cols_to_handle:
            if col != "meteo_rain_height":  # Exclude rain column for department-based medians
                self.department_medians[col] = (
                    X.groupby(self.department_col)[col].median()
                )
                self.global_means[col] = X[col].mean()  # Calculate global mean for fallback

        # Calculate mean rain by department and month
        if "meteo_rain_height" in self.cols_to_handle:
            X[self.date_col] = pd.to_datetime(X[self.date_col])
            X['month'] = X[self.date_col].dt.month
            self.rain_means_by_month_department = (
                X.groupby([self.department_col, 'month'])["meteo_rain_height"].mean()
            )
            print(f">> (Info) Rainfall means by department and month calculated.")

        print(f">> (Info) Medians and means successfully calculated.")
        return self

    def transform(self, X):
        print(f">> (Info) Filling missing values with calculated medians and means.")

        # Ensure all columns in cols_to_handle are numeric
        for col in self.cols_to_handle:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Fill missing values for columns grouped by department with fallback to global mean
        for col in self.cols_to_handle:
            if col != "meteo_rain_height":  # Skip rain column for department-based filling
                # Fill using department median
                X[col] = X[col].fillna(
                    X[self.department_col].map(self.department_medians[col])
                )
                # Fill remaining missing values using global mean
                X[col] = X[col].fillna(self.global_means[col])

        # Fill missing rain by average rain for the same month and department
        if "meteo_rain_height" in self.cols_to_handle:
            X[self.date_col] = pd.to_datetime(X[self.date_col])
            X['month'] = X[self.date_col].dt.month

            # Create a Series with aligned rain values
            rain_fill_values = X.set_index([self.department_col, 'month']).index.map(self.rain_means_by_month_department)

            # Ensure the result is a Series with the same index as X
            rain_fill_values = pd.Series(rain_fill_values, index=X.index)

            # Fill missing values in "meteo_rain_height"
            X["meteo_rain_height"] = X["meteo_rain_height"].fillna(rain_fill_values)

            X.drop(columns=['month'], inplace=True)  # Remove temporary column
        return X

# Clean pizzo


class CleanPizo(Transformer):
    '''
    Prepares and cleans the following features:

    - piezo_station_investigation_depth: Fill missing values with the mean of the department (piezo_station_department_code).
    - piezo_obtention_mode, piezo_status, piezo_qualification: One-hot encode, filling NaNs with the most frequent value.
    - piezo_measure_nature_code: Fill NaNs with "I", then one-hot encode.

    NEEDS : ["piezo_station_department_code"]
    INPUT : ['piezo_station_investigation_depth', 'piezo_obtention_mode', 'piezo_status', 'piezo_qualification', 'piezo_measure_nature_code']
    RETURNS : Cleaned and transformed columns one hot encoded 
    DROPS : None

    (ca renvoie bcp : 
    piezo_obtention_mode_Mode d'obtention inconnu',
       'piezo_obtention_mode_Valeur mesurée',
       'piezo_obtention_mode_Valeur reconstituée', 'piezo_status_Donnée brute',
       'piezo_status_Donnée contrôlée niveau 1',
       'piezo_status_Donnée contrôlée niveau 2',
       'piezo_status_Donnée interprétée', 'piezo_qualification_Correcte',
       'piezo_qualification_Incertaine', 'piezo_qualification_Incorrecte',
       'piezo_qualification_Non qualifié', 'piezo_measure_nature_code_0',
       'piezo_measure_nature_code_D', 'piezo_measure_nature_code_I',
       'piezo_measure_nature_code_N'])




    Example:
    cols = ['piezo_station_investigation_depth', 'piezo_obtention_mode', 'piezo_status', 'piezo_qualification', 'piezo_measure_nature_code']
    cleaner = CleanPizo(cols)

    '''

    def __init__(self, cols_to_handle, department_col="piezo_station_department_code"):
        # Initialize placeholders for the means, modes, and encoders
        self.department_col = department_col
        self.cols_to_handle = cols_to_handle
        self.department_means = {}  # For storing department-level means for numerical columns
        self.global_means = {}  # For storing global means for numerical columns
        self.column_modes = {}  # For storing the most frequent values (modes) for categorical columns
        self.one_hot_encoders = {}  # For storing one-hot encoders for categorical columns

    def fit(self, X, y=None):
        print(f">> (Info) Calculating means for numerical features and preparing for one-hot encoding.")

        # Handle piezo_station_investigation_depth: Calculate means
        depth_col = "piezo_station_investigation_depth"
        if depth_col in self.cols_to_handle:
            # Calculate department-level means
            self.department_means[depth_col] = X.groupby(self.department_col)[depth_col].mean()
            # Calculate global mean as fallback
            self.global_means[depth_col] = X[depth_col].mean()

        # Prepare for one-hot encoding and calculate modes for categorical columns
        for col in ['piezo_obtention_mode', 'piezo_status', 'piezo_qualification', 'piezo_measure_nature_code']:
            if col in self.cols_to_handle:
                # Calculate the most frequent value (mode) for the column
                self.column_modes[col] = X[col].mode()[0]

                # Special handling for piezo_measure_nature_code
                if col == 'piezo_measure_nature_code':
                    X[col] = X[col].astype(str).fillna('0')
                    X[col] = X[col].apply(lambda x: x if x in ['N', 'I', 'D', 'S'] else '0')

                # Store one-hot encoded column names
                self.one_hot_encoders[col] = pd.get_dummies(X[col], prefix=col, dtype=int).columns.tolist()

        print(f">> (Info) Fitting completed: Means, modes, and one-hot encoders prepared.")
        return self

    def transform(self, X):
        print(f">> (Info) Transforming data: Filling missing values and applying one-hot encoding.")

        # Handle piezo_station_investigation_depth
        depth_col = "piezo_station_investigation_depth"
        if depth_col in self.cols_to_handle:
            # Fill with department-level mean
            X[depth_col] = X[depth_col].fillna(
                X[self.department_col].map(self.department_means[depth_col])
            )
            # Fill remaining missing values with global mean
            X[depth_col] = X[depth_col].fillna(self.global_means[depth_col])
            print(f">> (Info) Missing values in {depth_col} filled with department means or global mean as fallback.")


        # Handle categorical columns with one-hot encoding and missing value handling
        for col in ['piezo_obtention_mode', 'piezo_status', 'piezo_qualification', 'piezo_measure_nature_code']:
            if col in self.cols_to_handle:
                # Fill missing values for piezo_measure_nature_code with "I"
                if col == 'piezo_measure_nature_code':
                    # Ensure all values are strings
                    X[col] = X[col].astype(str)

                    # Fill missing values with '0'
                    X[col] = X[col].fillna('0')

                    # Replace values not in ['N', 'I', 'D', 'S'] with '0'
                    X[col] = X[col].apply(lambda x: x if x in [
                                          'N', 'I', 'D', 'S'] else '0')
                else:
                    # Fill missing values with the most frequent value (mode)
                    X[col] = X[col].fillna(self.column_modes[col])

                # Apply one-hot encoding
                dummies = pd.get_dummies(X[col], prefix=col, dtype=int)

                # Ensure all one-hot columns exist, even if not present in test data
                for dummy_col in self.one_hot_encoders[col]:
                    if dummy_col not in dummies:
                        # Add missing column with default value 0
                        dummies[dummy_col] = 0

                # Align and concatenate
                # Ensure column order matches training
                dummies = dummies[self.one_hot_encoders[col]]
                X = pd.concat([X, dummies], axis=1)
                X.drop(columns=[col], inplace=True)
                print(
                    f">> (Info) One-hot encoding applied to {col} with missing values filled.")

        print(f">> (Info) Data transformation completed.")

        return X


class CleanTemp(Transformer):
    """
    Nettoyage des données relatives aux températures
    - Remplacement des valeurs manquantes de temp_avg en estimant à partir de temp_avg_threshold
    - idem pour temp_min_ground, à partir de temp_min
    - Au final, pour la température, on garde uniquement meteo_temperature_avg, meteo_temperature_min, meteo_temperature_max, meteo_temperature_min_ground
    Mettre ce Transformer avant TemperaturePressionTrans
    """

    def __init__(self):
        return

    def fit(self, X, y=None):
        X = X.copy()

        self.reglin_avg = LinearRegression().fit(
            X=pd.DataFrame(X.loc[
                X["meteo_temperature_avg_threshold"].notna(
                ) & X["meteo_temperature_avg"].notna(),
                "meteo_temperature_avg_threshold"
            ]),
            y=X.loc[
                X["meteo_temperature_avg_threshold"].notna(
                ) & X["meteo_temperature_avg"].notna(),
                "meteo_temperature_avg"
            ]
        )

        self.reglin_minground = LinearRegression().fit(
            X=pd.DataFrame(X.loc[
                X["meteo_temperature_min"].notna(
                ) & X["meteo_temperature_min_ground"].notna(),
                "meteo_temperature_min"
            ]),
            y=X.loc[
                X["meteo_temperature_min"].notna(
                ) & X["meteo_temperature_min_ground"].notna(),
                "meteo_temperature_min_ground"
            ]
        )

        return self

    def transform(self, X):
        X = X.copy()

        X.loc[
            X["meteo_temperature_avg"].isna(
            ) & X["meteo_temperature_avg_threshold"].notna(),
            "meteo_temperature_avg"
        ] = self.reglin_avg.predict(
            X=pd.DataFrame(X.loc[
                X["meteo_temperature_avg"].isna(
                ) & X["meteo_temperature_avg_threshold"].notna(),
                "meteo_temperature_avg_threshold"
            ])
        )

        X.loc[
            X["meteo_temperature_min_ground"].isna(
            ) & X["meteo_temperature_min"].notna(),
            "meteo_temperature_min_ground"
        ] = self.reglin_minground.predict(
            X=pd.DataFrame(X.loc[
                X["meteo_temperature_min_ground"].isna(
                ) & X["meteo_temperature_min"].notna(),
                "meteo_temperature_min"
            ])
        )

        X.drop(columns=['meteo_temperature_avg_threshold'], inplace=True, errors='ignore')

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
        # Partie 1 : supprimé les colonnes avec + de 60% de valeurs manquantes

        # Select only the specified columns
        relevant_cols = [col for col in self.columns if col in X.columns]

        # Calculate the threshold for missing values
        threshold = 0.6 * len(X)

        # Identify columns to drop within the relevant columns
        cols_to_drop = [
            col for col in relevant_cols if X[col].isna().sum() > threshold]

        # Drop the identified columns
        X = X.drop(columns=cols_to_drop)

        # Traitement des valeurs manquantes : moyenne sur le département à la meme date ou meme date si données manquantes

        for column in self.columns:
            if column in X.columns:
                # Check if the column contains NaN values
                if X[column].isna().sum() > 0:
                    # Fill NaN by department and date mean
                    moyennes_departement_date = X.groupby(
                        ['piezo_station_department_code', 'piezo_measurement_date'])[column].transform('mean')
                    X[column] = X[column].fillna(moyennes_departement_date)

                    # Step 3: Fill any remaining NaN by the mean of the date (ignoring the department)
                    moyennes_date = X.groupby('piezo_measurement_date')[
                        column].transform('mean')
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
        self.dist_to_meteo_threshold = dist_to_meteo_threshold

    def fit(self, X, y=None):
        if self.apply_threshold and self.dist_to_meteo_threshold is None:
            self.dist_to_meteo_threshold = X["distance_piezo_meteo"].quantile(
                0.95)
        return self

    def transform(self, X):
        X = X.copy()

        temp = X["meteo_longitude"].copy()
        X["meteo_longitude"] = X["meteo_latitude"].copy()
        X["meteo_latitude"] = temp

        if self.apply_threshold:
            X["near_meteo"] = (X["distance_piezo_meteo"] <=
                               self.dist_to_meteo_threshold).astype(float)
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
        # errors=ignore pour qu'il n y ait pas d'erreurs is la colonne n'existe pas
        X.drop(columns=drop_cols, inplace=True, errors='ignore')

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


class PrelevVol(Transformer):
    """Remplir les valeur manquantes des colonnes prelevement volume par la minimum de la colonne par commune de cette 

    NEEDS: ['piezo_station_commune_name', 'prelev_volume_0', 'prelev_volume_1', 'prelev_volume_2', 'prelev_other_volume_sum']
    INPUT : /
    RETURNS :
    DROP : piezo_station_commune_name
    """

    def __init__(self):
        self.columns = ['prelev_volume_0', 'prelev_volume_1',
                        'prelev_volume_2', 'prelev_other_volume_sum']

    def fit(self, X, y=None):
        self.min_vol = X.groupby('piezo_station_commune_name')[
            self.columns].min()
        self.min_vol = self.min_vol.fillna(0)
        print(
            f">> (INFO) missing values in columns {self.columns} are filled by the minimum of the column by commune")
        return self

    def transform(self, X):
        for col in self.columns:
            X[col] = X[col].fillna(
                X['piezo_station_commune_name'].map(self.min_vol[col]))
        X.drop(columns=['piezo_station_commune_name'], inplace=True)
        return X


class CleanHydro(Transformer):
    """
    Clean les données de la station hydrométrique
    - Valeurs aberrantes -> mean
    - Passe au log le resultat, en ajustant les valeurs négatives à 0

    NEEDS: ["hydro_observation_result_elab"]
    INPUTS: /
    RETURNS: ["hydro_observation_result_elab", "hydro_observation_log", "hydro_status_code", "hydro_qualification_code", "hydro_hydro_quantity_elab"]
    """

    def __init__(self):
        self.hydro_quantity_encoder = OneHotEncoder(
            max_categories=2, drop="first")
        return

    def fit(self, X, y=None):
        self.mean_without_outliers = X.loc[X["hydro_observation_result_elab"]
                                           < 1e8, "hydro_observation_result_elab"].mean()
        self.hydro_quantity_encoder.fit(
            pd.DataFrame(X["hydro_hydro_quantity_elab"]))
        return self

    def transform(self, X):
        X = X.copy()

        X.loc[X["hydro_observation_result_elab"] > 1e8,
              "hydro_observation_result_elab"] = self.mean_without_outliers

        X.loc[X["hydro_observation_result_elab"] <
              0, "hydro_observation_result_elab"] = 0
        X["hydro_observation_result_elab"] = X["hydro_observation_result_elab"]+1

        X["hydro_observation_log"] = X["hydro_observation_result_elab"].apply(
            np.log)

        X_encode = self.hydro_quantity_encoder.transform(
            pd.DataFrame(X["hydro_hydro_quantity_elab"])).toarray()
        X_encode_df = pd.DataFrame(
            X_encode, columns=self.hydro_quantity_encoder.get_feature_names_out(), index=X.index)
        X = pd.concat(
            [X.drop(columns=["hydro_hydro_quantity_elab"]), X_encode_df], axis=1)

        hydro_cols_to_drop = [
            "hydro_station_code",
            "hydro_observation_date_elab",
            "hydro_status_label",
            "hydro_method_code",
            "hydro_method_label",
            "hydro_qualification_label",
            "hydro_longitude",
            "hydro_latitude",
        ]
        X.drop(columns=hydro_cols_to_drop, inplace=True, errors="ignore")

        return X


class TimeTnx(Transformer):

    def __init__(
            self,
            columns: list[str],
            delta: int=5,
            clean: bool=True,
        ):

        self.columns = columns
        self.delta = delta
        self.clean = clean

    def fit(self, X, y=None):

        self.mean = X[self.columns].applymap(self.convert_minute).mean()

        return self

    def transform(self, X):

        X_ = X.copy()

        X_[self.columns] = X_[self.columns].applymap(self.convert_minute)

        def transform_row(row):
            if row['delta'] < self.delta:
                if row['meteo_time_tn'] <= 12*60 and row['meteo_time_tx'] <= 12*60:
                    row['meteo_time_tx'] = (row["meteo_time_tx"] + 12*60)%(24*60)
                elif 12*60 <= row['meteo_time_tn'] and row['meteo_time_tx'] <= 24:
                    row['meteo_time_tn'] = (row["meteo_time_tn"] + 12*60)%(24*60)
            return row

        if self.clean:
            delta_min_max = pd.DataFrame(abs(X_["meteo_time_tn"] - X_["meteo_time_tx"])%1440, columns=["delta"])
            X_ = pd.concat([X_, delta_min_max], axis=1)
            X_ = X_.apply(transform_row, axis=1)
            X_ = X_.drop(columns="delta")

        X_error = X_[(abs(X_["meteo_time_tn"] - X_["meteo_time_tx"])%1440 < 60*self.delta)]
        X_.iloc[X_error.index] = np.nan


        X_["meteo_time_tn"] = X_["meteo_time_tn"].fillna(300)
        X_["meteo_time_tx"] = X_["meteo_time_tx"].fillna(820)


        return X_
    
    def convert_minute(self, x: int) -> int:
        if pd.isna(x):  # Vérifie si x est NaN
            return np.nan
        
        x_str = str(int(x)).zfill(4)
        minutes = int(x_str[-2:])
        hours = int(x_str[:-2])

        return 60 * hours + minutes
