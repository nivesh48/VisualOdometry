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
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2,
                                   padding='same', activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(128, kernel_size=5, strides=2,
                                   padding='same', activation=tf.nn.relu)
        self.conv3 = layers.Conv2D(256, kernel_size=5, strides=2,
                                   padding='same', activation=tf.nn.relu)
        self.conv3_1 = layers.Conv2D(256, kernel_size=3, strides=1,
                                     padding='same', activation=tf.nn.relu)
        self.conv4 = layers.Conv2D(512, kernel_size=3, strides=2,
                                   padding='same', activation=tf.nn.relu)
        self.conv4_1 = layers.Conv2D(512, kernel_size=3, strides=1,
                                     padding='same', activation=tf.nn.relu)
        self.conv5 = layers.Conv2D(512, kernel_size=3, strides=2,
                                   padding='same', activation=tf.nn.relu)
        self.conv5_1 = layers.Conv2D(512, kernel_size=3, strides=1,
                                     padding='same', activation=tf.nn.relu)
        self.conv6 = layers.Conv2D(1024, kernel_size=3, strides=2,
                                   padding='same', activation=tf.nn.relu)
        self.reshape = keras.layers.Reshape((-1, 20 * 6 * 1024))

        rnn_cells = [layers.LSTMCell(1000) for _ in range(2)]
        stacked_lstm = layers.StackedRNNCells(rnn_cells)
        self.lstm_layer = layers.RNN(stacked_lstm)

        self.out = layers.Dense(6)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv5(x)
        x = self.conv5_1(x)
        x = self.conv6(x)
        x = self.reshape(x)
        with tf.device('/cpu:0'):
            x = self.lstm_layer(x)
            x = self.out(x)
        return x
