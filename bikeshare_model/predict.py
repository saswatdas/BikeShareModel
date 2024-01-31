import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.data_manager import pre_pipeline_preparation
from bikeshare_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bikeshare_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    errors = "" #Initialize
    #print(input_data)
    validated_data,errors = validate_inputs(input_df=pd.DataFrame(input_data))
    #validated_data = pd.DataFrame(input_data)
    validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = bikeshare_pipe.predict(validated_data)
    
    results = {"Requested: ": "Prediction for Winter Season, Nov 2012 on a working day with temp 6", "predictions": predictions,"version": _version}
    #print(results)
    #print('**********"Thank you!! *******')
    if not errors:
   
        predictions = bikeshare_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        

    return results

if __name__ == "__main__":

    #WORKING
    #data_in={'season':['winter'],'hr':['6am'],'holiday':['No'],'workingday':['Yes'],'weathersit':['Mist'],'temp':[6.1],'atemp':[3.0014],'hum':[49],'windspeed':[19.0012],'yr':[2012],'mnth':['November'], 'day_Friday':['0'], 'day_Monday':['1'], 'day_Saturday':['0'],'day_Sunday':[0],'day_Thursday':['0'],'day_Tuesday':['0'],'day_Wednesday':['0']}
    
    data_in={'dteday':['2012-05-11'],'season':['winter'],'hr':['6am'],'holiday':['No'],'weekday':['Mon'],'workingday':['Yes'],'weathersit':['Mist'],'temp':[6.1],'atemp':[3.0014],'hum':[49],'windspeed':[19.0012]}
    make_prediction(input_data=data_in)
