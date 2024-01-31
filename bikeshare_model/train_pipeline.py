import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt

from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    x=data.drop(config.model_config.target, axis=config.model_config.axis)
    y=data[config.model_config.target]
    
    #Order according t feature tag
    x=x.reindex(columns=config.model_config.features)
    
    #print (x.head(5))
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        x,y,
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

   # Pipeline fitting
    bikeshare_pipe.fit(X_train,y_train)
    y_pred = bikeshare_pipe.predict(X_test)
    #print('Gradient Boosting Regressor test mse: {}'.format(mean_squared_error(y_test, y_pred)))
    #print('Gradient Boosting Regressor test rmse: {}'.format(sqrt(mean_squared_error(y_test, y_pred))))
    #print('TEST DATA - R-squared: {}'.format(r2_score(y_test, y_pred)))
    #print('****************************************')

    # persist trained model
    save_pipeline(pipeline_to_persist= bikeshare_pipe)
    

    
if __name__ == "__main__":
    run_training()