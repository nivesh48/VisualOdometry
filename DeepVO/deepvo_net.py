# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:10
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DeepVONet(keras.Model):
    def __init__(self):
        super(DeepVONet, self).__init__()
        self.lstm1 = layers.LSTM(1000, return_sequences=True)
        self.lstm2 = layers.LSTM(1000)
        self.dropout = layers.Dropout(0.5)
        self.out = layers.Dense(6)

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        if training:
            x = self.dropout(x)
        x = self.out(x)
        return x
