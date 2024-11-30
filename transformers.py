
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

            X = X.fillna(self.mean[col])

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
        # Initialize placeholders for the medians and additional parameters
        self.department_col = department_col
        self.date_col = date_col
        self.meteo_group_means = None
        self.cols_to_handle = cols_to_handle
        self.department_medians = {}

    def fit(self, X, y=None):
        # Column names
        meteo = "meteo_rain_height"

        print(f">> (Info) Recuperations des moyennes des données INSEE par department")

        # Handle "meteo_rain_height"
        if meteo in self.cols_to_handle:

            X[self.date_col] = pd.to_datetime(X[self.date_col])
            X['month'] = X[self.date_col].dt.month
            self.meteo_group_means = (
                X.groupby([self.department_col, 'month'])[meteo]
                .mean()
                .reset_index()
                .rename(columns={meteo: 'mean_rain_height'})
            )

        # Handle all other columns (specified in cols_to_handle, excluding rain)
        for col in self.cols_to_handle:
            if col != meteo:

                X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)
                self.department_medians[col] = (
                    X.groupby(self.department_col)[col].median()
                )

        print(f">> (Info) Infos medianes Insee recupérees")

        return self

    def transform(self, X):
        # Column names
        meteo = "meteo_rain_height"

        # Handle "meteo_rain_height"
        if meteo in self.cols_to_handle:

            X[self.date_col] = pd.to_datetime(X[self.date_col])
            X['month'] = X[self.date_col].dt.month
            X = pd.merge(
                X,
                self.meteo_group_means,
                how='left',
                on=[self.department_col, 'month']
            )
            X[meteo] = X[meteo].fillna(X['mean_rain_height'])

            X.drop(columns=['mean_rain_height', 'month'], inplace=True)

        # Handle all other columns (specified in cols_to_handle, excluding rain)
        for col in self.cols_to_handle:
            if col != meteo:

                X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)
                X[col] = X[col].fillna(
                    X.groupby(self.department_col)[col].transform('median')
                )

        print(f">> (Info) Valeurs Manquantes comblées avec les Médianes.")

        return X
    
## Clean pizzo

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
        self.column_modes = {}  # For storing the most frequent values (modes) for categorical columns
        self.one_hot_encoders = {}  # For storing one-hot encoders for categorical columns

    def fit(self, X, y=None):
        print(f">> (Info) Calculating means for numerical features and preparing for one-hot encoding.")

        # Handle piezo_station_investigation_depth: Fill missing with mean of department
        depth_col = "piezo_station_investigation_depth"
        if depth_col in self.cols_to_handle:
            self.department_means[depth_col] = X.groupby(self.department_col)[depth_col].mean()

        # Prepare for one-hot encoding and calculate modes for categorical columns
        for col in ['piezo_obtention_mode', 'piezo_status', 'piezo_qualification', 'piezo_measure_nature_code']:
            if col in self.cols_to_handle:
                # Calculate the most frequent value (mode) for the column
                self.column_modes[col] = X[col].mode()[0]  # Store the most frequent value
                # Fill missing values for piezo_measure_nature_code with "I" during fitting
                if col == 'piezo_measure_nature_code':
                     # Ensure all values are strings
                    X[col] = X[col].astype(str)

                    # Fill missing values with '0'
                    X[col] = X[col].fillna('0')

                    # Replace values not in ['N', 'I', 'D', 'S'] with '0'
                    X[col] = X[col].apply(lambda x: x if x in ['N', 'I', 'D', 'S'] else '0')

                self.one_hot_encoders[col] = pd.get_dummies(X[col], prefix=col, dtype=int).columns.tolist()

        print(f">> (Info) Fitting completed: Means, modes, and one-hot encoders prepared.")

        return self

    def transform(self, X):
        print(f">> (Info) Transforming data: Filling missing values and applying one-hot encoding.")

        # Handle piezo_station_investigation_depth
        depth_col = "piezo_station_investigation_depth"
        if depth_col in self.cols_to_handle:
            X[depth_col] = X[depth_col].fillna(
                X.groupby(self.department_col)[depth_col].transform(lambda grp: grp.mean())
            )
            print(f">> (Info) Missing values in {depth_col} filled with department means.")

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
                    X[col] = X[col].apply(lambda x: x if x in ['N', 'I', 'D', 'S'] else '0')
                else:
                    # Fill missing values with the most frequent value (mode)
                    X[col] = X[col].fillna(self.column_modes[col])

                # Apply one-hot encoding
                dummies = pd.get_dummies(X[col], prefix=col, dtype=int)

                # Ensure all one-hot columns exist, even if not present in test data
                for dummy_col in self.one_hot_encoders[col]:
                    if dummy_col not in dummies:
                        dummies[dummy_col] = 0  # Add missing column with default value 0

                # Align and concatenate
                dummies = dummies[self.one_hot_encoders[col]]  # Ensure column order matches training
                X = pd.concat([X, dummies], axis=1)
                X.drop(columns=[col], inplace=True)
                print(f">> (Info) One-hot encoding applied to {col} with missing values filled.")

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
        print(X.columns)
        self.min_vol = X.groupby('piezo_station_commune_name')[
            self.columns].min()
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
        return

    def fit(self, X, y=None):
        self.mean_without_outliers = X.loc[X["hydro_observation_result_elab"]
                                           < 1e8, "hydro_observation_result_elab"].mean()
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
