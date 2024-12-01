from transformers import *
from sklearn.pipeline import Pipeline
import json

# Load the configuration file
with open("columns.json", "r") as f:
    columns = json.load(f)

col_yass = columns["col_yass"]
cols_yael_input = columns["cols_yael_input"]
cols_yael_need = columns["cols_yael_need"]
cols_lucien_need = columns["cols_lucien_need"]
cols_lucien_input = columns["cols_lucien_input"]
cols_mat = columns["cols_mat"]
pizo_cols = columns["pizo_cols"]
target = columns["target"]
columns_to_drop = columns["drop"]
prelev_flo = columns["prelev_flo"]
altitude_flo = columns["altitude_flo"]
meteo_time_flo = columns["meteo_time_flo"]
col_flo = altitude_flo + prelev_flo + meteo_time_flo


processing_pipeline = Pipeline(steps=[
    ("DropNaRate", DropNaRate(0.7)),
    ("MeteoTimeTnx", TimeTnx(delta=5, clean=False)),
    ("Prelevol", PrelevVol()),
    ("Prelevement", Prelev(columns=prelev_flo,
     usage_label_max_categories=4, mode_label_max_categories=4, scale=3)),
    ("CleanFeatures", CleanFeatures(cols_yael_input)),
    ("Altitude", AltitudeTrans(columns=[
     "piezo_station_altitude", "meteo_altitude"])),
    ('LatLong', CleanLatLon()),
    ('CleanTemp', CleanTemp()),
    ('Temp', TemperaturePressionTrans(columns=cols_lucien_input)),
    ('CleanHydro', CleanHydro()),
    ('CleanPizo',  CleanPizo(pizo_cols)),
    ('Dates', DateTransformer()),
    ('DropCols', DropCols(columns_to_drop))
])
