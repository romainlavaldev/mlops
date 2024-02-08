import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pickle



from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories,save_json



class ModelEvaluation:
    def __init__(self,config : ModelEvaluationConfig):
        self.config=config

    def eval_metrics(self, actual, pred):
        rmse=np.sqrt(mean_squared_error(actual,pred))
        mae=mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):

        #load datas
        with open(self.config.x_test_data_path, 'rb') as f:
            x_test = pickle.load(f)
        with open(self.config.y_test_data_path, 'rb') as f:
            y_test = pickle.load(f)

        x_test_without_time = x_test[:,:,1:].astype('float32')
        x_test_without_time[np.isnan(x_test_without_time)] = 0.0

        y_test_without_time = y_test[:,1:].astype('float32')
        y_test_without_time[np.isnan(y_test_without_time)] = 0.0

        with open(self.config.model_file, 'rb') as f:
            model = pickle.load(f)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            predicted_qualities= model.predict(x_test_without_time)
            (rmse,mae,r2)= self.eval_metrics(y_test_without_time, predicted_qualities)
            
            #Saving metrics as local
            scores={"rmse":rmse.astype('float'),"mae":mae.astype('float'),"r2":r2.astype('float')}

            print(scores)

            save_json(path=Path(self.config.metric_file_name),data=scores)

            mlflow.log_metric("rmse",rmse)
            mlflow.log_metric("r2",r2)
            mlflow.log_metric("mae",mae)