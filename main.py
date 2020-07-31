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
import tensorflow as tf
from utils.dataset import VisualOdometryDataLoader


# Custom loss function.
def custom_loss(y_, y, k, criterion):
    mse_position = criterion(y[:, :3], y_[:, :3])
    mse_orientation = criterion(y[:, 3:], y_[:, 3:])
    return mse_position + k * mse_orientation


def loss(model, x, y, k, criterion, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    return custom_loss(y_, y, k, criterion)


def grad(model, inputs, targets, k, criterion):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, k, criterion, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(flownet, model, config):
    print('Load Data...')
    dataset = VisualOdometryDataLoader(config['datapath'], 192, 640, config['bsize'])
    criterion = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(config['lr'])

    print('Training model...')
    train_loss_results = []
    for epoch in range(1, config['train_iter'] + 1):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for step, (batch_x, batch_y) in enumerate(dataset.dataset):
            with tf.device('/gpu:0'):
                x = flownet(batch_x)
            with tf.device('/cpu:0'):
                # Optimize the model
                loss_value, grads = grad(model, x, batch_y, config['k'], criterion)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
            print('Epoch {:03d}, \tStep {:04d}, \tLoss {:.3f}'.format(epoch, step, loss_value))
        train_loss_results.append(epoch_loss_avg.result())
        print('Epoch {:03d}, \tLoss: {:.3f}'.format(epoch, epoch_loss_avg.result()))
        model.save_weights(config['checkpoint_path'] + '/' + config['train'] + '/cp.ckpt')

    print('Plot loss...')
    fig, ax = plt.subplots()
    ax.plot(train_loss_results)

    ax.set(xlabel='Epoch', ylabel='Loss', title='Training Metrics ' + config['train'])
    ax.grid()

    fig.savefig('loss_' + config['train'] + '.png')
    plt.show()


def test(flownet, model, config):
    print('Load Data...')
    dataset = VisualOdometryDataLoader(config['datapath'], 192, 640, config['bsize'], True)

    model.load_weights(config['checkpoint_path'] + '/' + config['train'] + '/cp.ckpt')

    x, x_pred = 0.0, 0.0
    y, y_pred = 0.0, 0.0
    z, z_pred = 0.0, 0.0

    X, X_pred = [], []
    Y, Y_pred = [], []
    Z, Z_pred = [], []

    X.append(x)
    X_pred.append(x_pred)

    Y.append(y)
    Y_pred.append(y_pred)

    Z.append(z)
    Z_pred.append(z_pred)

    for step, (input_img, y_true) in enumerate(dataset.dataset):
        print('Sequence: ' + str(step))
        with tf.device('/gpu:0'):
            flow = flownet(tf.expand_dims(input_img, 0))
        with tf.device('/cpu:0'):
            pred = model(flow).numpy()[0]
        y_true = y_true.numpy()

        x += y_true[0]
        x_pred += pred[0]

        y += y_true[1]
        y_pred += pred[1]

        z += y_true[2]
        z_pred += pred[2]

        X.append(x)
        X_pred.append(x_pred)
        Y.append(y)
        Y_pred.append(y_pred)
        Z.append(z)
        Z_pred.append(z_pred)

    print('Plot trajectory...')
    fig, ax = plt.subplots()
    ax.plot(X, Y)
    ax.plot(X_pred, Y_pred)
    ax.set(
        xlabel='Distance', ylabel='Distance',
        title='Comparison ' + config['train'])
    ax.grid()
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
        'mode': 'train',
        'datapath': 'D:\EduardoTayupanta\Documents\Librerias\dataset',
        'bsize': 8,
        'lr': 0.001,
        'train_iter': 200,
        'checkpoint_path': './checkpoints',
        'k': 100,
        'train': 'deepvo'
    }

    flownet = FlowNet()

    if config['mode'] == 'train':
        if config['train'] == 'deepvo':
            with tf.device('/cpu:0'):
                deepvonet = DeepVONet()
            train(flownet, deepvonet, config)
        elif config['train'] == 'magicvo':
            with tf.device('/cpu:0'):
                magicvonet = MagicVONet()
            train(flownet, magicvonet, config)
    elif config['mode'] == 'test':
        if config['train'] == 'deepvo':
            with tf.device('/cpu:0'):
                deepvonet = DeepVONet()
            test(flownet, deepvonet, config)
        elif config['train'] == 'magicvo':
            with tf.device('/cpu:0'):
                magicvonet = MagicVONet()
            test(flownet, magicvonet, config)


if __name__ == "__main__":
    main()
