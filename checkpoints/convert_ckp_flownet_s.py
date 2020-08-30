# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:11
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FlowNetS:
    def __init__(self, height, width):
        conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=(height, width, 6),
                              name='conv1')
        leaky_relu_1 = layers.LeakyReLU(0.1)

        conv2 = layers.Conv2D(128, kernel_size=5, strides=2, padding='same', name='conv2')
        leaky_relu_2 = layers.LeakyReLU(0.1)

        conv3 = layers.Conv2D(256, kernel_size=5, strides=2, padding='same', name='conv3')
        leaky_relu_3 = layers.LeakyReLU(0.1)

        conv3_1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='conv3_1')
        leaky_relu_3_1 = layers.LeakyReLU(0.1)

        conv4 = layers.Conv2D(512, kernel_size=3, strides=2, padding='same', name='conv4')
        leaky_relu_4 = layers.LeakyReLU(0.1)

        conv4_1 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', name='conv4_1')
        leaky_relu_4_1 = layers.LeakyReLU(0.1)

        conv5 = layers.Conv2D(512, kernel_size=3, strides=2, padding='same', name='conv5')
        leaky_relu_5 = layers.LeakyReLU(0.1)

        conv5_1 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', name='conv5_1')
        leaky_relu_5_1 = layers.LeakyReLU(0.1)

        conv6 = layers.Conv2D(1024, kernel_size=3, strides=2, padding='same', name='conv6')
        leaky_relu_6 = layers.LeakyReLU(0.1)

        self.model = keras.Sequential([
            conv1,
            leaky_relu_1,
            conv2,
            leaky_relu_2,
            conv3,
            leaky_relu_3,
            conv3_1,
            leaky_relu_3_1,
            conv4,
            leaky_relu_4,
            conv4_1,
            leaky_relu_4_1,
            conv5,
            leaky_relu_5,
            conv5_1,
            leaky_relu_5_1,
            conv6,
            leaky_relu_6,
        ])


path = 'D:\EduardoTayupanta\Documents\Librerias\FlowNet\checkpoints\FlowNetS'
file = 'flownet-S.ckpt-0'
trained_checkpoint_prefix = os.path.join(path, file)
net = FlowNetS(192, 640)

names = [
    'conv1',
    'conv2',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6'
]

weights = {}
biases = {}
for name in names:
    weights['FlowNetS/' + name + '/weights:0'] = None
    biases['FlowNetS/' + name + '/biases:0'] = None

loaded_graph = tf.Graph()
with tf.compat.v1.Session(graph=loaded_graph) as sess:
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    variables_names = [v.name for v in tf.compat.v1.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        if k == 'FlowNetS/conv1/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv2/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv3/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv3_1/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv4/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv4_1/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv5/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv5_1/weights:0':
            weights[k] = v
        elif k == 'FlowNetS/conv6/weights:0':
            weights[k] = v

        if k == 'FlowNetS/conv1/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv2/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv3/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv3_1/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv4/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv4_1/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv5/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv5_1/biases:0':
            biases[k] = v
        elif k == 'FlowNetS/conv6/biases:0':
            biases[k] = v

i = 0
for name_weights, name_biases in zip(weights, biases):
    l = []
    l.append(weights[name_weights])
    l.append(biases[name_biases])
    net.model.layers[i].set_weights(l)
    i += 2

net.model.save('flownet_s.h5')
