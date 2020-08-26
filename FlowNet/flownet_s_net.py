# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:10
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
from tensorflow import keras


class FlowNet(keras.Model):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.flow_net = keras.models.load_model('checkpoints/flownet_s.h5')

    def call(self, inputs, **kwargs):
        x = self.flow_net(inputs)
        return x
