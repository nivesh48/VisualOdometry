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
from utils.dataset import VisualOdometryDataLoader


def test(flownet, model, config):
    print('Load Data...')
    print('=' * 50)
    dataset = VisualOdometryDataLoader(config['datapath'], 192, 640, config['bsize'], test=True,
                                       sequence_test=config['sequence'])

    # model.load_weights(config['checkpoint_path'] + '/' + config['test'] + '/ckpt-10')
    optimizer = None
    if config['test'] == 'deepvo':
        optimizer = tf.keras.optimizers.SGD(config['lr'], 0.9, True)
    elif config['test'] == 'magicvo':
        optimizer = tf.keras.optimizers.Adagrad(config['lr'])
    elif config['test'] == 'poseconvgru':
        optimizer = tf.keras.optimizers.Adam(config['lr'])

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, config['checkpoint_path'] + '/' + config['test'], max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print('=' * 50)

    print('Test model {}...'.format(config['test'].upper()))
    print('=' * 50)

    T = np.eye(4)
    gtT = np.eye(4)

    estimatedCameraTraj = np.empty([len(dataset) + 1, 3])
    gtCameraTraj = np.empty([len(dataset) + 1, 3])

    estimatedCameraTraj[0] = np.zeros([1, 3])
    gtCameraTraj[0] = np.zeros([1, 3])

    estimatedFrame = 0
    gtFrame = 0

    for step, (batch_x, batch_y) in enumerate(dataset.dataset):
        print('Sequence: ' + str(step))
        with tf.device('/gpu:0'):
            flow = flownet(batch_x)
        with tf.device('/cpu:0'):
            batch_predict_pose = model(flow)

        for pred in batch_predict_pose.numpy():
            R = dataset.eulerAnglesToRotationMatrix(pred[3:])
            t = pred[:3].reshape(3, 1)
            T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

            # With respect to the first frame
            T_abs = np.dot(T, T_r)
            # Update the T matrix till now.
            T = T_abs

            # Get the origin of the frame (i+1), ie the camera center
            estimatedCameraTraj[estimatedFrame + 1] = np.transpose(T[0:3, 3])
            estimatedFrame += 1

        for gt in batch_y.numpy():
            R = dataset.eulerAnglesToRotationMatrix(gt[3:])
            t = gt[:3].reshape(3, 1)
            gtT_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)

            # With respect to the first frame
            gtT_abs = np.dot(gtT, gtT_r)
            # Update the T matrix till now.
            gtT = gtT_abs

            # Get the origin of the frame (i+1), ie the camera center
            gtCameraTraj[gtFrame + 1] = np.transpose(gtT[0:3, 3])
            gtFrame += 1

    np.save('output/' + config['sequence'] + '_GroundTruth.npy', gtCameraTraj)
    np.save('output/' + config['test'] + '_' + config['sequence'] + '_Estimated.npy', estimatedCameraTraj)

    # Plot the estimated and groundtruth trajectories
    x_gt = gtCameraTraj[:, 0]
    z_gt = gtCameraTraj[:, 2]

    x_est = estimatedCameraTraj[:, 0]
    z_est = estimatedCameraTraj[:, 2]

    fig, ax = plt.subplots(1)
    ax.plot(x_gt, z_gt, 'c', label="ground truth")
    ax.plot(x_est, z_est, 'm', label="estimated")
    ax.legend()
    plt.show()


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
        'lr': 0.001,  # deepvo - magicvo
        # 'lr': 0.0001, # poseconvgru
        'checkpoint_path': './checkpoints',
        'test': 'magicvo',
        'sequence': '03'
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
    elif config['test'] == 'poseconvgru':
        with tf.device('/cpu:0'):
            poseconvgru = PoseConvGRUNet()
        test(flownet, poseconvgru, config)


if __name__ == "__main__":
    main()
