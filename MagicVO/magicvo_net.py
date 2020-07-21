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
        lstm_fw = layers.LSTM(1000)
        lstm_bw = layers.LSTM(1000, go_backwards=True)
        self.bi_lstm = layers.Bidirectional(lstm_fw, backward_layer=lstm_bw)
        self.dense = layers.Dense(256, activation=tf.nn.relu)
        self.out = layers.Dense(6)

    def call(self, inputs, **kwargs):
        x = self.bi_lstm(inputs)
        x = self.dense(x)
        x = self.out(x)
        return x
