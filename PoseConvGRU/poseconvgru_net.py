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


class PoseConvGRUNet(keras.Model):
    def __init__(self):
        super(PoseConvGRUNet, self).__init__()
        self.max_pooling = layers.MaxPool2D(2, strides=2)
        self.reshape = layers.Reshape((-1, 5 * 1 * 1024))
        self.gru = layers.GRU(3)
        self.dense_1 = layers.Dense(4096)
        self.leaky_relu_1 = layers.LeakyReLU(0.1)
        self.dense_2 = layers.Dense(1024)
        self.leaky_relu_2 = layers.LeakyReLU(0.1)
        self.dense_3 = layers.Dense(128)
        self.leaky_relu_3 = layers.LeakyReLU(0.1)
        self.out = layers.Dense(6)

    def call(self, inputs, is_training=False):
        x = self.max_pooling(inputs)
        x = self.reshape(x)
        x = self.gru(x)
        x = self.dense_1(x)
        x = self.leaky_relu_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        x = self.dense_3(x)
        x = self.leaky_relu_3(x)
        x = self.out(x)
        return x
