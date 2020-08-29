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
import tensorflow as tf
from utils.dataset import VisualOdometryDataLoader


# Custom loss function.
def custom_loss(y_, y, k, criterion):
    mse_position = criterion(y[:, :3], y_[:, :3])
    mse_orientation = criterion(y[:, 3:], y_[:, 3:])
    return mse_position + k * mse_orientation


def loss(model, x, y, k, criterion, is_training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, is_training=is_training)
    return custom_loss(y_, y, k, criterion)


def grad(model, inputs, targets, k, criterion):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, k, criterion, is_training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(flownet, model, config):
    print('Load Data...')
    print('=' * 50)
    train_dataset = VisualOdometryDataLoader(config['datapath'], 192, 640, config['bsize'])
    val_dataset = VisualOdometryDataLoader(config['datapath'], 192, 640, config['bsize'], val=True)

    criterion = tf.keras.losses.MeanSquaredError()
    optimizer = None
    if config['train'] == 'deepvo':
        optimizer = tf.keras.optimizers.SGD(config['lr'], 0.9, True)
    elif config['train'] == 'magicvo':
        optimizer = tf.keras.optimizers.Adagrad(config['lr'])
    elif config['train'] == 'poseconvgru':
        optimizer = tf.keras.optimizers.Adam(config['lr'])

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, config['checkpoint_path'] + '/' + config['train'], max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print('=' * 50)

    print('Training model {}...'.format(config['train'].upper()))
    print('=' * 50)
    train_loss_results = []
    val_loss_results = []
    for epoch in range(1, config['train_iter'] + 1):
        epoch_train_loss_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()

        print('[=', end='')
        for step, (batch_x, batch_y) in enumerate(train_dataset.dataset):
            with tf.device('/gpu:0'):
                x = flownet(batch_x)
            with tf.device('/cpu:0'):
                # Optimize the model
                loss_value, grads = grad(model, x, batch_y, config['k'], criterion)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_train_loss_avg.update_state(loss_value)

        print('=', end='')
        for step, (batch_x, batch_y) in enumerate(val_dataset.dataset):
            with tf.device('/gpu:0'):
                x = flownet(batch_x)
            with tf.device('/cpu:0'):
                loss_value = loss(model, x, batch_y, config['k'], criterion, is_training=False)
            epoch_val_loss_avg.update_state(loss_value)

        train_loss_results.append(epoch_train_loss_avg.result())
        val_loss_results.append(epoch_val_loss_avg.result())

        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print('=' * 50)
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("Training Loss: {:.10f} \tValidation Loss: {:.10f}".format(epoch_train_loss_avg.result(),
                                                                             epoch_val_loss_avg.result()))
            print('=' * 50)
        else:
            print('] Epoch {:03d}, \tTraining Loss: {:.10f}, \tValidation Loss: {:.10f}'.format(epoch,
                                                                                                epoch_train_loss_avg.result(),
                                                                                                epoch_val_loss_avg.result()))

    print('Plot loss...')
    print('=' * 50)
    fig, ax = plt.subplots()
    ax.plot(train_loss_results, 'b--', label='Training')
    ax.plot(val_loss_results, 'r', label='Validation')

    ax.set(xlabel='Epoch', ylabel='Loss', title='Training Metrics -' + config['train'].upper())
    ax.grid()

    fig.savefig('loss_' + config['train'] + '.png')
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
        'lr': 0.001,
        # 'lr': 0.0001,
        'train_iter': 200,
        'checkpoint_path': './checkpoints',
        'k': 100,
        'train': 'deepvo'
    }

    flownet = FlowNet()

    if config['train'] == 'deepvo':
        with tf.device('/cpu:0'):
            deepvonet = DeepVONet()
        train(flownet, deepvonet, config)
    elif config['train'] == 'magicvo':
        with tf.device('/cpu:0'):
            magicvonet = MagicVONet()
        train(flownet, magicvonet, config)
    elif config['train'] == 'poseconvgru':
        with tf.device('/cpu:0'):
            poseconvgru = PoseConvGRUNet()
        train(flownet, poseconvgru, config)


if __name__ == "__main__":
    main()
