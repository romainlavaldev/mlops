import pandas as pd
import os
from mlProject import logger
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout, Bidirectional
import pickle



from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config

    def load_model(self, input_shape, n_steps):
        regressor = Sequential()
        regressor.add(Bidirectional(LSTM(units=150, return_sequences=True, input_shape = input_shape ) ))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units= 100 , return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units= 80 , return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units= 100))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = n_steps, activation='linear'))
        regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
        
        return regressor


    def train(self):
        #load datas
        with open(self.config.x_train_data_path, 'rb') as f:
            x_train = pickle.load(f)
        with open(self.config.y_train_data_path, 'rb') as f:
            y_train = pickle.load(f)
        model = self.load_model((x_train.shape[1], 1), 4)

        x_train_without_time = x_train[:,:,1:].astype('float32')
        x_train_without_time[np.isnan(x_train_without_time)] = 0.0

        y_train_without_time = y_train[:,1:].astype('float32')
        y_train_without_time[np.isnan(y_train_without_time)] = 0.0

        if not os.path.exists(self.config.root_dir + '/' + self.config.model_file):
            model.fit(x_train_without_time, y_train_without_time, epochs=20, batch_size=64 )

            with open(self.config.root_dir + '/' + self.config.model_file, 'wb') as f:
                pickle.dump(model, f)
