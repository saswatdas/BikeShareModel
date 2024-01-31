import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeathersitImputer,tempOutlierHandler,windspeedOutlierHandler,humOutlierHandler
from bikeshare_model.processing.features import Mapper
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score

bikeshare_pipe = Pipeline([

    ('weathersit_imputation', WeathersitImputer(variables=config.model_config.weathersit_var)),
    #('weekday_imputation', weekdayImputer(variables='day')),

    ##==========Mapper======##
    ('map_yr',Mapper(config.model_config.mapyear_var, config.model_config.year_mappings)),
    ('map_mnth',Mapper(config.model_config.month_var, config.model_config.month_mappings)),
    ('map_holiday',Mapper(config.model_config.holiday_var, config.model_config.holiday_mappings)),
    ('map_workingday',Mapper(config.model_config.workingday_var, config.model_config.workingday_mappings)),
    ('map_season',Mapper(config.model_config.season_var, config.model_config.season_mappings)),
    ('map_weathersit',Mapper(config.model_config.weathersit_var, config.model_config.weathersit_mappings)),
    ('map_hr',Mapper(config.model_config.hour_var,config.model_config.hour_mappings)),

    # Transformation of temperature,windspeed and hum columns
    ('tempOutlierHandler', tempOutlierHandler(variables=config.model_config.temp_var)),
    ('windspeedOutlierHandler', windspeedOutlierHandler(variables=config.model_config.windspeed_var)),
    ('humOutlierHandler', humOutlierHandler(variables=config.model_config.hum_var)),

    # scale & model (Gradient Boost Regressor)
    ('scaler', StandardScaler()),
    ('model_gb', ensemble.GradientBoostingRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth, learning_rate=config.model_config.learning_rate, criterion=config.model_config.criterion))
])
