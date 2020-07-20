# Visual of Odometry Implementation in TensorFlow 2.2

This repository is an implementation of the following architectures:

- [DeepVO: Towards end-to-end visual odometry with deep Recurrent Convolutional Neural Networks](https://ieeexplore.ieee.org/document/7989236)
- [MagicVO: An End-to-End Hybrid CNN and Bi-LSTM Method for Monocular Visual Odometry](https://ieeexplore.ieee.org/document/8753500)

The code uses the FlowNetS pre-trained model [FlowNet: Learning Optical Flow with Convolutional Networks](https://ieeexplore.ieee.org/document/7410673).

Inside the `main.py` file is the asdas variable that serves as the configuration for the training.

- `mode` code execution mode, such as to `train` or to `predict`.
- `datapath` path where the dataset is stored.
- `bsize` size of batch size.
- `lr` learning rate value for SGD optimizer.
- `momentum` momentum value for SGD optimizer.
- `train_iter` number of epoch for training.
- `checkpoint_path` path where the checkpoint are stored.
- `k` default value for loss function.
- `train` model `DeepVO` or `MagicVO` to be trained or predicted.

## Download weights

To download the weights of the models, download and place them in the `checkpoints` folder, where the download instructions are located.

## Training

For training, the [KITTI]() Visual Odometry dataset has been used, you can change the training sequences in the file `utils/dataset.py`. For example, the following variable `self.sequences = ['00', '02', '08', '09']` has been used for sequences `00`, `02`, `08` and `09`, which are the most extensive.

The structure containing the dataset must agree to the following:

`<path where the dataset has been stored>\dataset`

    -->\poses

        --> \00.txt

        --> \01.txt

        ...

    -->\sequences

        --> \00

        --> \01

        ...

- Run the `main.py` file with changes to the `config` variable for DeepVO model training

`
config = {
    'mode': 'train',
    'datapath': 'D:\EduardoTayupanta\Documents\Librerias\dataset',
    'bsize': 8,
    'lr': 0.001,
    'momentum': 0.99,
    'train_iter': 20,
    'checkpoint_path': './checkpoints',
    'k': 100,
    'train': 'deepvo'
}
`

- Run the `main.py` file with changes to the `config` variable for MagicVO model training

`
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
`

## Prediction
