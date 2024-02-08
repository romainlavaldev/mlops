import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import pickle



class PredictPipeline:
    def __init__(self):
        with open(Path('artifacts/model_trainer/trained_model'), 'rb') as f:
            self.model = pickle.load(f)
    
    def get_data_batch(self, date):
        datas = pd.read_csv('artifacts/data_ingestion/cryptos.csv')

        return datas[datas.Time < date].tail(30).drop('Time', axis=1).to_numpy()[np.newaxis, ...]

    def get_crypto_preds_id(self, crypto_code):
        if crypto_code == 'BTC':
            return 0
        if crypto_code == 'ETH':
            return 1
        if crypto_code == 'DOGE':
            return 2
        if crypto_code == 'ADA':
            return 3

    def predict(self,crypto, date):
        prediction = self.model.predict(self.get_data_batch(date))
        print(prediction)
        return prediction[0][self.get_crypto_preds_id(crypto)]
