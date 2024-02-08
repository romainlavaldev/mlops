import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config
    
    def split_sequences(self, sequences, n_step):
        X, y = list(), list()

        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_step
            # check if we are beyond the sequence
            if end_ix > len(sequences)-1:
                break

            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_file)

        X, y = self.split_sequences(data.to_numpy(), 30)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        with open(self.config.root_dir+'/X_train', 'wb') as f:
            pickle.dump(X_train, f)
        with open(self.config.root_dir+'/X_test', 'wb') as f:
            pickle.dump(X_test, f)
        with open(self.config.root_dir+'/y_train', 'wb') as f:
            pickle.dump(y_train, f)
        with open(self.config.root_dir+'/y_test', 'wb') as f:
            pickle.dump(y_test, f)

        logger.info("Splited data into training and test sets")
        logger.info(X_train.shape)
        logger.info(X_test.shape)

        print(X_train.shape)
        print(X_test.shape)
        