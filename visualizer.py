# -*- coding: utf-8 -*-
"""
Created by etayupanta at 7/1/2020 - 16:11
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_route(ax, x_gt, z_gt, x_est, z_est, method, sequence):
    ax.scatter(x_gt[0], z_gt[0], label='Sequence start', marker='s', color='k')
    ax.plot(x_gt, z_gt, 'k', label='Ground truth', linewidth=2.5)
    colors = ['b', 'g', 'm', 'y', 'r']
    c = np.random.choice(len(colors), size=1, replace=False)
    ax.plot(x_est, z_est, colors[c[0]] + '' + '-.', label=method, linewidth=3.5)
    ax.legend(loc='upper left', fontsize='x-large')
    ax.grid(b=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax.set_title('Visual Odometry - Sequence ' + sequence, fontdict={'fontsize': 20})
    plt.show()


def comparison(ax, x_gt, z_gt, x_est, z_est, title, method):
    ax.scatter(x_gt[0], z_gt[0], label='Sequence start', marker='s', color='k')
    ax.plot(x_gt, z_gt, 'k', label='Ground truth', linewidth=2.5)
    index = 0
    colors = ['b', 'g', 'm', 'y', 'r']
    descriptions = ['--', '-.', ':']
    c = np.random.choice(len(colors), size=len(method), replace=False)
    d = np.random.choice(len(descriptions), size=len(method), replace=False)
    for x, y in zip(x_est, z_est):
        ax.plot(x, y, colors[c[index]] + '' + descriptions[d[index]], label=method[index], linewidth=3.5)
        index += 1
    ax.legend(loc='upper left', fontsize='x-large')
    ax.grid(b=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax.set_title(title, fontdict={'fontsize': 20})
    plt.show()


def error(axe, i, abs_error, title, method, colors, c, descriptions, d):
    ax = axe[i]
    index = 0
    for abs_error in abs_error:
        ax.plot(abs_error, colors[c[index]] + '' + descriptions[d[index]], label=method[index], linewidth=3.5)
        index += 1
    ax.legend(loc='upper left', fontsize='x-large')
    ax.grid(b=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax.set_title(title, fontdict={'fontsize': 20})


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def visualizer(config):
    gtCameraTraj = np.load('output/' + config['sequence'] + '_GroundTruth.npy')
    deepvo = np.load('output/deepvo_' + config['sequence'] + '_Estimated.npy')
    magicvo = np.load('output/magicvo_' + config['sequence'] + '_Estimated.npy')
    poseconvgru = np.load('output/poseconvgru_' + config['sequence'] + '_Estimated.npy')

    gt_rt = np.array([[1, 2, 3, 4, 5, 6], [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]])
    deepvo_rt = np.array([[1.1, 2.1, 3.1, 4.1, 5.1, 6.1], [1.2, 2.2, 3.2, 4.2, 5.2, 6.2]])
    magicvo_rt = np.array([[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], [1.2, 2.2, 3.2, 4.2, 5.2, 6.2]])
    poseconvgru_rt = np.array([[0.9, 1.9, 2.9, 3.9, 4.9, 5.9], [1.2, 2.2, 3.2, 4.2, 5.2, 6.2]])

    # Plot the estimated and groundtruth trajectories
    x_gt = gtCameraTraj[:, 0]
    z_gt = gtCameraTraj[:, 2]

    x_deepvo = deepvo[:, 0]
    z_deepvo = deepvo[:, 2]

    x_magicvo = magicvo[:, 0]
    z_magicvo = magicvo[:, 2]

    x_poseconvgru = poseconvgru[:, 0]
    z_poseconvgru = poseconvgru[:, 2]

    fig, ax = plt.subplots(1, figsize=(12, 12))
    plot_route(ax,
               x_gt, z_gt,
               x_deepvo, z_deepvo,
               'DeepVO',
               config['sequence'])

    fig, ax = plt.subplots(1, figsize=(12, 12))
    plot_route(ax,
               x_gt, z_gt,
               x_magicvo, z_magicvo,
               'MagicVO',
               config['sequence'])

    fig, ax = plt.subplots(1, figsize=(12, 12))
    plot_route(ax,
               x_gt, z_gt,
               x_poseconvgru, z_poseconvgru,
               'PoseConvGRU',
               config['sequence'])

    x_est = [x_deepvo, x_magicvo, x_poseconvgru]
    z_est = [z_deepvo, z_magicvo, z_poseconvgru]
    fig, ax = plt.subplots(1, figsize=(12, 12))
    comparison(ax,
               x_gt, z_gt,
               x_est, z_est,
               'Visual Odometry / Sequence ' + config['sequence'],
               ['DeepVO', 'MagicVO', 'PoseConvGRU'])

    abs_error_deepvo = abs(gt_rt - deepvo_rt)
    abs_error_magicvo = abs(gt_rt - magicvo_rt)
    abs_error_poseconvgru = abs(gt_rt - poseconvgru_rt)

    x_abs_error = [abs_error_deepvo[:, 0], abs_error_magicvo[:, 0], abs_error_poseconvgru[:, 0]]
    y_abs_error = [abs_error_deepvo[:, 1], abs_error_magicvo[:, 1], abs_error_poseconvgru[:, 1]]
    z_abs_error = [abs_error_deepvo[:, 2], abs_error_magicvo[:, 2], abs_error_poseconvgru[:, 2]]

    roll_abs_error = [abs_error_deepvo[:, 3], abs_error_magicvo[:, 3], abs_error_poseconvgru[:, 3]]
    pitch_abs_error = [abs_error_deepvo[:, 4], abs_error_magicvo[:, 4], abs_error_poseconvgru[:, 4]]
    yaw_abs_error = [abs_error_deepvo[:, 5], abs_error_magicvo[:, 5], abs_error_poseconvgru[:, 5]]

    fig, ax = plt.subplots(1, 3, figsize=(36, 12))
    fig.suptitle('Absolut Error / Translational - Sequence ' + config['sequence'], fontsize=24)
    method = ['DeepVO', 'MagicVO', 'PoseConvGRU']
    colors = ['b', 'g', 'm', 'y', 'r']
    descriptions = ['--', '-.', ':']
    c = np.random.choice(len(colors), size=len(method), replace=False)
    d = np.random.choice(len(descriptions), size=len(method), replace=False)
    error(ax,
          0,
          x_abs_error,
          'X',
          method,
          colors, c,
          descriptions, d)
    error(ax,
          1,
          y_abs_error,
          'Y',
          method,
          colors, c,
          descriptions, d)
    error(ax,
          2,
          z_abs_error,
          'Z',
          method,
          colors, c,
          descriptions, d)
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(36, 12))
    fig.suptitle('Absolut Error / Rotational - Sequence ' + config['sequence'], fontsize=24)
    error(ax,
          0,
          roll_abs_error,
          'Roll',
          method,
          colors, c,
          descriptions, d)
    error(ax,
          1,
          pitch_abs_error,
          'Pitch',
          method,
          colors, c,
          descriptions, d)
    error(ax,
          2,
          yaw_abs_error,
          'Yaw',
          method,
          colors, c,
          descriptions, d)
    plt.show()

    t_rmse_deepvo = calculate_rmse(gt_rt[:, :3], deepvo_rt[:, :3])
    r_rmse_deepvo = calculate_rmse(gt_rt[:, 3:], deepvo_rt[:, 3:])

    t_rmse_magicvo = calculate_rmse(gt_rt[:, :3], magicvo_rt[:, :3])
    r_rmse_magicvo = calculate_rmse(gt_rt[:, 3:], magicvo_rt[:, 3:])

    t_rmse_poseconvgru = calculate_rmse(gt_rt[:, :3], poseconvgru_rt[:, :3])
    r_rmse_poseconvgru = calculate_rmse(gt_rt[:, 3:], poseconvgru_rt[:, 3:])

    data = {'Translational RMSE': [t_rmse_deepvo, t_rmse_magicvo, t_rmse_poseconvgru],
            'Rotational RMSE': [r_rmse_deepvo, r_rmse_magicvo, r_rmse_poseconvgru]}

    rmse = pd.DataFrame(data, index=['DeepVO', 'MagicVO', 'PoseConvGRU'],
                        columns=['Translational RMSE', 'Rotational RMSE'])
    print(rmse)


def main():
    config = {
        'sequence': '05'
    }
    visualizer(config)


if __name__ == "__main__":
    main()
