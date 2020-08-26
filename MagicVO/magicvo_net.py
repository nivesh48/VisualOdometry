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


class MagicVONet(keras.Model):
    def __init__(self):
        super(MagicVONet, self).__init__()
        self.reshape = keras.layers.Reshape((-1, 10 * 3 * 1024))
        lstm_fw = layers.LSTM(1000, dropout=0.5)
        lstm_bw = layers.LSTM(1000, dropout=0.5, go_backwards=True)
        self.bi_lstm = layers.Bidirectional(lstm_fw, backward_layer=lstm_bw)
        self.dense = layers.Dense(256)
        self.leaky_relu = layers.LeakyReLU(0.1)
        self.out = layers.Dense(6)

    def call(self, inputs, is_training=False):
        x = self.reshape(inputs)
        x = self.bi_lstm(x)
        x = self.dense(x)
        x = self.leaky_relu(x)
        x = self.out(x)
        return x
