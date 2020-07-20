# Visual of Odometry Implementation in TensorFlow 2.2

This repository is an implementation of the following architectures:

- [DeepVO: Towards end-to-end visual odometry with deep Recurrent Convolutional Neural Networks](https://ieeexplore.ieee.org/document/7989236)
- [MagicVO: An End-to-End Hybrid CNN and Bi-LSTM Method for Monocular Visual Odometry](https://ieeexplore.ieee.org/document/8753500)

The code uses the FlowNetS pre-trained model [FlowNet: Learning Optical Flow with Convolutional Networks](https://ieeexplore.ieee.org/document/7410673).

## Download weights

To download the weights of the models, download and place them in the `checkpoints` folder, where the download instructions are located.

## Training

For training, the [KITTI]() Visual Odometry dataset has been used, you can change the training sequences in the file `utils/dataset.py`. For example, the following variable `self.sequences = ['00', '02', '08', '09']` has been used for sequences 00, 02, 08 and 09, which are the most extensive.

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

## Prediction
