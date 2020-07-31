# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:11
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
from FlowNet.flownet_s_net import FlowNet
from DeepVO.deepvo_net import DeepVONet
from MagicVO.magicvo_net import MagicVONet
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.dataset import VisualOdometryDataLoader


def test(flownet, model, config):
    print('Load Data...')
    dataset = VisualOdometryDataLoader(config['datapath'], 192, 640, config['bsize'], True)

    model.load_weights(config['checkpoint_path'] + '/' + config['test'] + '/cp.ckpt')

    for step, (input_img, y_true) in enumerate(dataset.dataset):
        print('Sequence: ' + str(step))
        with tf.device('/gpu:0'):
            flow = flownet(input_img)
        with tf.device('/cpu:0'):
            batch_predict_pose = model(flow).numpy()[0]


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    config = {
        'datapath': 'D:\EduardoTayupanta\Documents\Librerias\dataset',
        'bsize': 8,
        'checkpoint_path': './checkpoints',
        'test': 'deepvo'
    }

    flownet = FlowNet()

    if config['test'] == 'deepvo':
        with tf.device('/cpu:0'):
            deepvonet = DeepVONet()
        test(flownet, deepvonet, config)
    elif config['test'] == 'magicvo':
        with tf.device('/cpu:0'):
            magicvonet = MagicVONet()
        test(flownet, magicvonet, config)


if __name__ == "__main__":
    main()
