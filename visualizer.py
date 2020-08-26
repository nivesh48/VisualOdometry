# -*- coding: utf-8 -*-
"""
Created by etayupanta at 7/1/2020 - 16:11
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
import matplotlib.pyplot as plt
import numpy as np


def visualizer(config):
    gtCameraTraj = np.load('output/' + config['sequence'] + '_GroundTruth.npy')
    estimatedCameraTraj = np.load('output/' + config['test'] + '_' + config['sequence'] + '_Estimated.npy')

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
    config = {
        'test': 'deepvo',
        'sequence': '03'
    }
    visualizer(config)


if __name__ == "__main__":
    main()
