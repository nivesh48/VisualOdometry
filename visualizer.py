# -*- coding: utf-8 -*-
"""
Created by etayupanta at 7/1/2020 - 16:11
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
import math
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


def plot_one_route(ax, x_gt, z_gt, sequence):
    ax.scatter(x_gt[0], z_gt[0], label='Sequence start', marker='s', color='k')
    ax.plot(x_gt, z_gt, 'k', label='Ground truth', linewidth=2.5)
    ax.legend(loc='upper left', fontsize='x-large')
    ax.grid(b=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax.set_title('Sequence ' + sequence, fontdict={'fontsize': 20})
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
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    # assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z], dtype=np.float32)


def translational_rotational(Rt):
    values = []
    for element in Rt[1:]:
        rt = element.reshape((3, 4))
        R = rt[:, 0:3]
        angles = rotationMatrixToEulerAngles(R)
        t = rt[:, 3]
        values.append([t[0], t[1], t[2], angles[0], angles[1], angles[2]])
    return np.array(values, dtype=np.float32)


def visualizer(config):
    gtCameraTraj = np.load('output/' + config['sequence'] + '_GroundTruth.npy')
    deepvo = np.load('output/deepvo_' + config['sequence'] + '_Estimated.npy')
    magicvo = np.load('output/magicvo_' + config['sequence'] + '_Estimated.npy')
    poseconvgru = np.load('output/poseconvgru_' + config['sequence'] + '_Estimated.npy')

    rp_gt_rt = np.load('output/' + config['sequence'] + '_RelativePoseGroundTruth.npy')
    rp_deepvo_rt = np.load('output/deepvo_' + config['sequence'] + '_RelativePoseEstimated.npy')
    rp_magicvo_rt = np.load('output/magicvo_' + config['sequence'] + '_RelativePoseEstimated.npy')
    rp_poseconvgru_rt = np.load('output/poseconvgru_' + config['sequence'] + '_RelativePoseEstimated.npy')

    gt_rt = translational_rotational(rp_gt_rt)
    deepvo_rt = translational_rotational(rp_deepvo_rt)
    magicvo_rt = translational_rotational(rp_magicvo_rt)
    poseconvgru_rt = translational_rotational(rp_poseconvgru_rt)

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


def visualizer_gt(config):
    gtCameraTraj = np.load('output/' + config['sequence'] + '_GroundTruth.npy')
    x_gt = gtCameraTraj[:, 0]
    z_gt = gtCameraTraj[:, 2]

    fig, ax = plt.subplots(1, figsize=(12, 12))
    plot_one_route(ax,
               x_gt, z_gt,
               config['sequence'])


def main():
    config = {
        'sequence': '03'
    }
    visualizer(config)
    # visualizer_gt(config)


if __name__ == "__main__":
    main()
