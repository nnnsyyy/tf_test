""" Basic functions for tf pipeline """

import tensorflow as tf
import pandas as pd
from . import shared


class Base:
    def __init__(self):
        self.width = shared.IMAGE_WIDTH
        self.length = round(shared.IMAGE_WIDTH / shared.WL_RATIO)
        # self.length = shared.IMAGE_LENGTH

    def load_data_csv(self, _label, _is_url, _train, _test):
        # Create a local copy of the training set.
        # train_path holds the pathname: ~/.keras/datasets/iris_training.csv
        if(_is_url):
            train_path = tf.keras.utils.get_file(
                fname=_train.split('/')[-1], origin=_train)
            test_path = tf.keras.utils.get_file(
                fname=_test.split('/')[-1], origin=_test)
        else:
            train_path = _train
            test_path = _test

        train_set = pd.read_csv(filepath_or_buffer=train_path,
                                names=shared.CSV_COLUMN_NAMES,
                                # ignore the first row of the CSV file.
                                header=0
                                )
        train_features, train_label = train_set, train_set.pop(_label)
        test_set = pd.read_csv(filepath_or_buffer=test_path,
                               names=shared.CSV_COLUMN_NAMES,
                               # ignore the first row of the CSV file.
                               header=0
                               )
        test_features, test_label = test_set, test_set.pop(_label)

        return (train_features, train_label, test_features, test_label)

    def get_feature_cols(self, _train_feature):
        feature_cols = []
        for key in _train_feature.keys():
            feature_cols.append(tf.feature_column.numeric_column(key=key))
        return feature_cols
