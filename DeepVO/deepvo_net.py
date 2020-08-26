# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:10
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
from tensorflow import keras
from tensorflow.keras import layers


class DeepVONet(keras.Model):
    def __init__(self):
        super(DeepVONet, self).__init__()
        self.reshape = keras.layers.Reshape((-1, 10 * 3 * 1024))
        self.lstm1 = layers.LSTM(1000, dropout=0.5, return_sequences=True)
        self.lstm2 = layers.LSTM(1000, dropout=0.5)
        self.dropout = layers.Dropout(0.5)
        self.out = layers.Dense(6)

    def call(self, inputs, is_training=False):
        x = self.reshape(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dropout(x, is_training)
        x = self.out(x)
        return x
