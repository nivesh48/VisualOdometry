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
from PoseConvGRU.poseconvgru_net import PoseConvGRUNet
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import time
from utils.dataset import VisualOdometryDataLoader


class RealTimeVisualizer:
    def __init__(self, config):
        self.flow_net = FlowNet()

        self.sequence = config['sequence']
        self.data_path = config['data_path']
        self.bsize = config['bsize']
        self.checkpoint_path = config['checkpoint_path']
        self.test = config['test']

        self.dataset = None

        if config['test'] == 'deepvo':
            self.optimizer = tf.keras.optimizers.SGD(config['lr'], 0.9, True)
            self.color = 'b'
            self.description = '--'
            with tf.device('/cpu:0'):
                self.model = DeepVONet()
        elif config['test'] == 'magicvo':
            self.optimizer = tf.keras.optimizers.Adagrad(config['lr'])
            self.color = 'y'
            self.description = '-.'
            with tf.device('/cpu:0'):
                self.model = MagicVONet()
        elif config['test'] == 'poseconvgru':
            self.optimizer = tf.keras.optimizers.Adam(config['lr'])
            self.color = 'r'
            self.description = ':'
            with tf.device('/cpu:0'):
                self.model = PoseConvGRUNet()

        print('=' * 50)
        print('Visualizer model {}\nSequence {}'.format(self.test.upper(), self.sequence))
        print('=' * 50)

        self.restore_checkpoint()
        self.load_data()

    def restore_checkpoint(self):
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path + '/' + self.test, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        print('=' * 50)

    def load_data(self):
        print('Load Data...')
        print('=' * 50)
        self.dataset = VisualOdometryDataLoader(self.data_path, 192, 640, self.bsize, test=True,
                                                sequence_test=self.sequence)

    def show(self):
        T = np.eye(4)
        gtT = np.eye(4)

        pred_x = np.array([0])
        pred_z = np.array([0])

        truth_x = np.array([0])
        truth_z = np.array([0])

        plt.ion()
        figure, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(0, 0, label='Sequence start', marker='s', color='k')
        ax.plot(truth_x, truth_z, 'k', label='Ground truth', linewidth=2.5)
        ax.plot(pred_x, pred_z, self.color + '' + self.description, label=self.test, linewidth=3.5)
        ax.legend(loc='lower right', fontsize='x-large')
        ax.grid(b=True, which='major', color='#666666', linestyle='-')
        ax.minorticks_on()
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax.set_title('Visual Odometry - Sequence ' + self.sequence, fontdict={'fontsize': 20})
        ax.set_xlabel('X[m]', fontdict={'fontsize': 16})
        ax.set_ylabel('Z[m]', fontdict={'fontsize': 16})
        ax.set_xlim(-320, 260)
        ax.set_ylim(-160, 400)

        for step, (batch_x, batch_y) in enumerate(self.dataset.dataset):
            print('=' * 50)
            print('Step: ' + str(step))
            time_flow_net_start = time()
            with tf.device('/gpu:0'):
                flow = self.flow_net(batch_x)
            time_flow_net_end = time()
            time_model_start = time()
            with tf.device('/cpu:0'):
                batch_predict_pose = self.model(flow)
            time_model_end = time()

            print('Inference time FlowNet: {:.10f} [s]'.format(time_flow_net_end - time_flow_net_start))
            print('Inference time {}: {:.10f} [s]'.format(self.test.upper(), time_model_end - time_model_start))
            print('-' * 50)
            print('Error X[m]\t\t\tError Z[m]')

            poses = zip(batch_y.numpy(), batch_predict_pose.numpy())
            for y, pred in poses:
                R = self.dataset.eulerAnglesToRotationMatrix(y[3:])
                t = y[:3].reshape(3, 1)
                gtT_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)
                gtT_abs = np.dot(gtT, gtT_r)
                gtT = gtT_abs

                truth = np.transpose(gtT[0:3, 3])
                truth_x = np.append(truth_x, truth[0])
                truth_z = np.append(truth_z, truth[2])

                R = self.dataset.eulerAnglesToRotationMatrix(pred[3:])
                t = pred[:3].reshape(3, 1)
                T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)
                T_abs = np.dot(T, T_r)
                T = T_abs

                prediction = np.transpose(T[0:3, 3])
                pred_x = np.append(pred_x, prediction[0])
                pred_z = np.append(pred_z, prediction[2])

                print('{:.6f}\t\t\t{:.6f}'.format(abs(truth[0] - prediction[0]), abs(truth[2] - prediction[2])))

                ax.plot(pred_x, pred_z, self.color + '' + self.description, label=self.test, linewidth=3.5)
                ax.plot(truth_x, truth_z, 'k', label='Ground truth', linewidth=2.5)
                plt.show()
                plt.pause(0.0001)


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    config = {
        'data_path': 'D:\EduardoTayupanta\Documents\Librerias\dataset',
        'bsize': 8,
        'lr': 0.001,  # deepvo - magicvo
        # 'lr': 0.0001,  # poseconvgru
        'checkpoint_path': './checkpoints',
        'test': 'magicvo',
        'sequence': '05'
    }

    visualizer = RealTimeVisualizer(config)
    visualizer.show()


if __name__ == "__main__":
    main()
