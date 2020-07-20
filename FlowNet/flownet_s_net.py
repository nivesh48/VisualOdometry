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


class FlowNet(keras.Model):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.flownet = tf.keras.models.load_model('checkpoints/flownet_s.h5')
        self.flownet.trainable = False
        self.reshape = keras.layers.Reshape((-1, 20 * 6 * 1024))

    def call(self, inputs, **kwargs):
        x = self.flownet(inputs)
        x = self.reshape(x)
        return x
