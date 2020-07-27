# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:11
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
import cv2
from DeepVO.deepvo_net import DeepVONet
from MagicVO.magicvo_net import MagicVONet
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.dataset import VisualOdometryDataLoader


# Custom loss function.
def custom_loss(y_pred, y_true, k, criterion):
    mse_position = criterion(y_true[:, :3], y_pred[:, :3])
    mse_orientation = criterion(y_true[:, 3:], y_pred[:, 3:])
    return mse_position + k * mse_orientation


def run_optimization(model, x, y, k, criterion, optimizer):
    with tf.GradientTape() as g:
        # Forward pass.
        pred = model(x, is_training=True)
        # Compute loss.
        loss = custom_loss(pred, y, k, criterion)

    # Variables to update, i.e. trainable variables.
    trainable_variables = model.trainable_variables

    with tf.device('/cpu:0'):
        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


def train_model(dataset, model, config, criterion, optimizer, epoch):
    loss = 0.0
    for step, (batch_x, batch_y) in enumerate(dataset):
        run_optimization(
            model,
            batch_x,
            batch_y,
            config['k'],
            criterion,
            optimizer)

        pred = model(batch_x)

        loss = custom_loss(
            pred,
            batch_y,
            config['k'],
            criterion
        ).numpy()

        print('Epoch {}, \t Step: {}, \t Loss: {}'.format(epoch, step, loss))
    return loss


def train(model, config):
    print('Load Data...')
    dataset = VisualOdometryDataLoader(
        config['datapath'], 384, 1280, config['bsize'])

    criterion = tf.keras.losses.MeanSquaredError()
    if config['train'] == 'deepvo':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config['lr'],
            momentum=config['momentum'],
            nesterov=True
        )
    elif config['train'] == 'magicvo':
        optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=config['lr'],
        )

    print('Training model...')
    total_loss = []
    for epoch in range(1, config['train_iter'] + 1):
        loss = train_model(
            dataset.dataset,
            model,
            config,
            criterion,
            optimizer,
            epoch)
        total_loss.append(loss)

        model.save_weights(config['checkpoint_path'] +
                           '/' + config['train'] + '/cp.ckpt')

    print('Plot loss...')
    fig, ax = plt.subplots()
    ax.plot(range(len(total_loss)), total_loss)

    ax.set(xlabel='Epoch Number', ylabel='Loss Magnitude',
           title='Loss per epoch model ' + config['train'])
    ax.grid()

    fig.savefig('loss_' + config['train'] + '.png')
    plt.show()


def decode_img(img):
    dim = (1280, 384)
    image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    image = image.astype('float32')
    return image


def get_input(paths):
    print(paths)
    img1 = cv2.imread(paths[0])
    img2 = cv2.imread(paths[1])
    img1 = decode_img(img1)
    img2 = decode_img(img2)
    return np.concatenate((img1, img2), axis=-1)


def test(model, config):
    print('Load Data...')
    dataset = VisualOdometryDataLoader(
        config['datapath'], 384, 1280, config['bsize'], True)

    model.load_weights(config['checkpoint_path'] +
                       '/' + config['train'] + '/cp.ckpt')

    images_stacked, odometries = dataset.images_stacked, dataset.odometries
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

    for index in range(len(odometries)):
        input_img = get_input(images_stacked[index])
        pred = model(np.array([input_img])).numpy()[0]
            
        x += odometries[index][0]
        x_pred += pred[0]

        y += odometries[index][1]
        y_pred += pred[1]

        z += odometries[index][2]
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
    ax.set(xlabel='Distance', ylabel='Distance',
           title='Comparison ' + config['train'])
    ax.grid()
    plt.show()

    # input_img = get_input(images_stacked[0])
    # pred = model(np.array([input_img])).numpy()[0]
    # print(odometries[0], pred)
    # print(odometries[0], pred[0])
    # print(odometries[0], pred.numpy()[0])
    # print(odometries[0], pred.numpy()[1])
    # print(odometries[0], pred.numpy()[2])


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    config = {
        'mode': 'test',
        'datapath': 'D:\EduardoTayupanta\Documents\Librerias\dataset',
        'bsize': 8,
        'lr': 0.001,
        'momentum': 0.99,
        'train_iter': 20,
        'checkpoint_path': './checkpoints',
        'k': 100,
        'train': 'deepvo'
    }

    deepvonet = DeepVONet()
    magicvonet = MagicVONet()

    if config['mode'] == 'train':
        if config['train'] == 'deepvo':
            train(deepvonet, config)
        elif config['train'] == 'magicvo':
            train(magicvonet, config)
    elif config['mode'] == 'test':
        if config['train'] == 'deepvo':
            test(deepvonet, config)
        elif config['train'] == 'magicvo':
            test(magicvonet, config)


if __name__ == "__main__":
    main()
