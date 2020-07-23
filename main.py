# -*- coding: utf-8 -*-
"""
Created by etayupanta at 6/30/2020 - 21:11
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
from DeepVO.deepvo_net import DeepVONet
from FlowNet.flownet_s_net import FlowNet
from MagicVO.magicvo_net import MagicVONet
import matplotlib.pyplot as plt
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

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


def train_model(dataset, flownet, model, config, criterion, optimizer, epoch):
    loss = 0.0
    for step, (batch_x, batch_y) in enumerate(dataset.dataset):
        with tf.device('/gpu:0'):
            flow = flownet(batch_x)
        with tf.device('/cpu:0'):
            run_optimization(
                model,
                flow,
                batch_y,
                config['k'],
                criterion,
                optimizer)

            pred = model(flow)

        loss = custom_loss(
            pred,
            batch_y,
            config['k'],
            criterion
        ).numpy()

        print('Epoch {}, \t Step: {}, \t Loss: {}'.format(epoch, step, loss))
    return loss


def train(flownet, model, config):
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
            dataset,
            flownet,
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


def test(flownet, model, config):
    print(path)


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
        'momentum': 0.99,
        'train_iter': 20,
        'checkpoint_path': './checkpoints',
        'k': 100,
        'train': 'magicvo'
    }

    deepvonet = DeepVONet()
    flownet = FlowNet()
    magicvonet = MagicVONet()

    if config['mode'] == 'train':
        if config['train'] == 'deepvo':
            train(flownet, deepvonet, config)
        elif config['train'] == 'magicvo':
            train(flownet, magicvonet, config)
    elif config['mode'] == 'test':
        if config['train'] == 'deepvo':
            test(flownet, deepvonet, config)
        elif config['train'] == 'magicvo':
            test(flownet, magicvonet, config)


if __name__ == "__main__":
    main()
