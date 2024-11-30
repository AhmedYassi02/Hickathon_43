import pandas as pd
import numpy as np
import optuna
import catboost as cb
from xgboost import XGBClassifier
import pickle
from transformers import *

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')
columns_to_drop = [
    "piezo_station_department_name",
    "piezo_station_update_date",
    "piezo_station_commune_code_insee",
    "piezo_station_pe_label",
    "piezo_station_bdlisa_codes",
    "piezo_station_bss_code",
    "piezo_station_bss_id",
    "piezo_bss_code",
    "piezo_measurement_date",
    "piezo_producer_name",
    "piezo_measure_nature_code",
    "meteo_name",
    "meteo_id",
    "meteo_latitude",
    "meteo_longitude",
    "hydro_station_code",
    "hydro_method_code",
    "hydro_method_label",
    "hydro_qualification_label",
    "prelev_structure_code_0",
    "prelev_structure_code_2",
    "prelev_structure_code_0",
    "prelev_commune_code_insee_0",
    "piezo_station_department_code",

    "meteo_DRR",
    "meteo_temperature_min_ground",
    "meteo_temperature_min_50cm",
    "meteo_pressure_avg",
    "meteo_pression_maxi",
    "meteo_wind_speed_avg_2m",
    "meteo_wind_max_2m",
    "meteo_wind_direction_max_inst_2m",
    "meteo_time_wind_max_2m",
    "meteo_wetting_duration",
    "meteo_sunshine_duration",
    "meteo_radiation",
    "meteo_radiation_direct",
    "meteo_sunshine_%",
    "meteo_radiation_IR",
    "meteo_radiation_UV_max",
    "meteo_cloudiness",
    "meteo_cloudiness_height",
    "meteo_if_snow",
    "meteo_if_fog",
    "meteo_if_thunderstorm",
    "meteo_if_sleet",
    "meteo_if_hail",
    "meteo_if_dew",
    "meteo_if_black_ice",
    "meteo_if_snow_ground",
    "meteo_if_frost",
    "meteo_if_smoke",
    "meteo_if_mist",
    "meteo_if_lightning",
    "meteo_evapotranspiration_Monteith",
    "meteo_radiation_UV",
    "meteo_snow_height",
    "meteo_snow_thickness_max",
    "meteo_snow_thickness_6h"]


# columns_to_drop += ajouts_drop_yael
path_src_dataset = Path("./data/src/X_train_Hi5.csv")

col_yass = ['meteo_date'] + ['prelev_volume_0', 'prelev_volume_1',
                             'prelev_volume_2', 'prelev_other_volume_sum', 'piezo_station_commune_name']
# Altitude
altitude_flo = ["piezo_station_altitude", "meteo_altitude"]  # ORDRE 1
prelev_flo = ["prelev_volume_0", "prelev_usage_label_0", "prelev_volume_obtention_mode_label_0", "prelev_volume_1", "prelev_usage_label_1",
              "prelev_volume_obtention_mode_label_1", "prelev_volume_2", "prelev_usage_label_2", "prelev_volume_obtention_mode_label_2"]
col_flo = altitude_flo + prelev_flo
# Insee & rain "CleanFeatures"
cols_yael_input = ['insee_%_agri', 'meteo_rain_height', 'insee_pop_commune',
                   'insee_med_living_level', 'insee_%_ind', 'insee_%_const']
cols_yael_need = ["piezo_station_department_code", "meteo_date"]

# Temperature
cols_lucien_need = ['piezo_station_department_code', 'piezo_measurement_date']
cols_lucien_input = ['meteo_temperature_avg', 'meteo_temperature_min',
                     'meteo__pressure_saturation_avg', 'meteo_temperature_max']
# Lat Long
cols_mat = ["distance_piezo_meteo", 'piezo_station_longitude', 'piezo_station_latitude', 'meteo_latitude', 'meteo_longitude', "meteo_temperature_avg", "meteo_temperature_avg_threshold",
            "meteo_temperature_min", "meteo_temperature_max", "meteo_temperature_min_ground", "hydro_observation_result_elab", "hydro_status_code", "hydro_qualification_code", "hydro_hydro_quantity_elab"]


# Clean pizo
pizo_cols = ['piezo_station_investigation_depth', 'piezo_obtention_mode', 'piezo_status',
             'piezo_qualification', 'piezo_measure_nature_code', 'piezo_station_department_code']

# target
target = "piezo_groundwater_level_category"

columns_to_keep = col_yass + cols_yael_input + cols_yael_need + col_flo + \
    cols_lucien_need + cols_lucien_input + cols_mat + [target] + pizo_cols

# Out folders
out_folder_dataset = Path("./data/cleaned")
# Create the folder if it doesn't exist
out_folder_dataset.mkdir(parents=True, exist_ok=True)
out_folder_config = Path("./data/processed/pipelines")
out_folder_config.mkdir(parents=True, exist_ok=True)

# Load the CSV file with only the relevant columns
# SI on veut charger moins de lignes : ajouter --> ,nrows=10e4)
df = pd.read_csv(path_src_dataset, usecols=columns_to_keep)
df = df.drop_duplicates()


X = df.drop(columns=target)

# Mapping du target
mapping = {'Very Low': 0, 'Low': 1, 'Average': 2, 'High': 3, 'Very High': 4}
y = df[target].map(mapping)


# Test-val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Apply the transformers selected
processing_pipeline = Pipeline(steps=[
    ("DropNaRate", DropNaRate(0.7)),
    ('PrelevVol', PrelevVol()),
    ("Prelevement", Prelev(columns=col_flo, usage_label_max_categories=4,
     mode_label_max_categories=4, scale=1)),
    ("CleanFeatures", CleanFeatures(cols_yael_input)),
    ("Altitude", AltitudeTrans(columns=[
     "piezo_station_altitude", "meteo_altitude"])),
    ('LatLong', CleanLatLon()),
    ('CleanTemp', CleanTemp()),
    ('Temp', TemperaturePressionTrans(columns=cols_lucien_input)),
    ('CleanHydro', CleanHydro()),
    ('CleanPizo',  CleanPizo(pizo_cols)),
    ('Dates', DateTransformer()),
    ('DropCols', DropCols(columns_to_drop)),
    ('scaler', PartialStandardScaler(columns='all'))
])


processed_X_train = processing_pipeline.fit_transform(X_train)
processed_X_val = processing_pipeline.transform(X_val)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators_gb", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate_gb", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth_gb", 2, 5),
        "objective": "multi:softmax",
        "device": "gpu",
    }

    model = XGBClassifier(**params)

    model.fit(processed_X_train, y_train)
    predictions = model.predict(processed_X_val)
    f1 = f1_score(y_val, predictions, average='weighted')
    return f1


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3_gb",
        study_name="XGB_1",
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=200)
