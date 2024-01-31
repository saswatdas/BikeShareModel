import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer


from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


##  Pre-Pipeline Preparation

def get_year_and_month(dataframe) -> pd.DataFrame:

    df = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year
    df['mnth'] = df['dteday'].dt.month_name()
    df['day'] = df['dteday'].dt.day_name()

    return df

# Treat 'day' column as a Categorical variable, perform one-hot encoding
def EncodeDay(dataframe):

    df = dataframe.copy()
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[['day']])
    encoded_weekday = encoder.transform(df[['day']])

    # Get encoded feature names
    enc_wkday_features = encoder.get_feature_names_out(['day'])
    # Append encoded weekday features to X
    df[enc_wkday_features] = encoded_weekday

    return df


def dropunusedcolumns(dataframe)-> pd.DataFrame:
    unused_colms = ['dteday', 'weekday', 'casual', 'day', 'registered']   # unused columns to be removed
    df = dataframe.copy()
    #drop the unwanted columns (ignore error if column does not exist)
    df=df.drop(columns=unused_colms, errors='ignore')

    return df

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    df = data_frame.copy()
  
    #derive the year and month directly from the date (as 'weekday' column is erroneous)
    df = get_year_and_month(df)
   
    #OneHotEncoding to be done for the 'day' column
    df=EncodeDay(df)
    
    #Opost encoding check for 'day' column
    df=postencodingcheck(df)
    
    #print(df.head(5))
    # drop unnecessary feature columns
    df = dropunusedcolumns (df)
    
    return df

def postencodingcheck(dataframe) -> pd.DataFrame:
    df = dataframe.copy()
    if 'day_Sunday' not in df: df['day_Sunday'] = 0
    if 'day_Monday' not in df: df['day_Monday'] = 0
    if 'day_Tuesday' not in df: df['day_Tuesday'] = 0
    if 'day_Wednesday' not in df: df['day_Wednesday'] = 0
    if 'day_Thursday' not in df: df['day_Thursday'] = 0
    if 'day_Friday' not in df: df['day_Friday'] = 0
    if 'day_Saturday' not in df: df['day_Saturday'] = 0
    
    return df

def dropunusedcolumns(dataframe) -> pd.DataFrame:
    unused_colms = ['dteday', 'weekday', 'casual', 'day', 'registered']   # unused columns to be removed
    df = dataframe.copy()
    #drop the unwanted columns
    df=df.drop(columns=unused_colms,errors='ignore')

    return df

def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
