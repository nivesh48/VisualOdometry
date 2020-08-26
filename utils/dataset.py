# -*- coding: utf-8 -*-
"""
Created by etayupanta at 7/1/2020 - 16:11
__author__ = 'Eduardo Tayupanta'
__email__ = 'eduardotayupanta@outlook.com'
"""

# Import Libraries:
import math
import numpy as np
import os
import tensorflow as tf


class VisualOdometryDataLoader:
    def __init__(self, datapath, height, width, batch_size, test=False, val=False, sequence_test='01'):
        self.base_path = datapath
        if test or val:
            self.sequences = [sequence_test]
        else:
            # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            # self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            self.sequences = ['00', '02', '04', '06', '08', '10']

        self.size = 0
        self.sizes = []
        self.poses = self.load_poses()
        self.width = width
        self.height = height

        images_stacked, odometries = self.get_data()
        dataset = tf.data.Dataset.from_tensor_slices((images_stacked, odometries))

        if test:
            dataset = dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(len(images_stacked))
            dataset = dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.dataset = dataset

    def decode_img(self, img):
        image = tf.image.decode_png(img, channels=3)
        image = image[..., ::-1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.height, self.width])
        return image

    def load_image(self, filename, odometry):
        img1 = tf.io.read_file(filename[0])
        img2 = tf.io.read_file(filename[1])
        img1 = self.decode_img(img1)
        img2 = self.decode_img(img2)
        img = tf.concat([img1, img2], -1)
        return img, odometry

    def load_poses(self):
        all_poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.base_path, 'poses/', sequence + '.txt')) as f:
                poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                all_poses.append(poses)

                self.size = self.size + len(poses)
                self.sizes.append(len(poses))
        return all_poses

    def get_image_paths(self, sequence, index):
        image_path = os.path.join(self.base_path, 'sequences', sequence, 'image_2', '%06d' % index + '.png')
        return image_path

    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        assert (self.isRotationMatrix(R))
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

    def eulerAnglesToRotationMatrix(self, theta):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta[0]), -np.sin(theta[0])],
                        [0, np.sin(theta[0]), np.cos(theta[0])]
                        ])
        R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                        [0, 1, 0],
                        [-np.sin(theta[1]), 0, np.cos(theta[1])]
                        ])
        R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                        [np.sin(theta[2]), np.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def matrix_rt(self, p):
        return np.vstack([np.reshape(p.astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])

    def get_data(self):
        images_paths = []
        odometries = []
        for index, sequence in enumerate(self.sequences):
            for i in range(self.sizes[index] - 1):
                images_paths.append([self.get_image_paths(sequence, i), self.get_image_paths(sequence, i + 1)])
                pose1 = self.matrix_rt(self.poses[index][i])
                pose2 = self.matrix_rt(self.poses[index][i + 1])
                pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
                R = pose2wrt1[0:3, 0:3]
                t = pose2wrt1[0:3, 3]
                angles = self.rotationMatrixToEulerAngles(R)
                odometries.append(np.concatenate((t, angles)))
        return np.array(images_paths), np.array(odometries)

    def __len__(self):
        return self.size - len(self.sequences)


def main():
    path = "D:\EduardoTayupanta\Documents\Librerias\dataset"
    dataset = VisualOdometryDataLoader(path, 192, 640, 32)
    for element in dataset.dataset.as_numpy_iterator():
        for index in range(len(element[0])):
            img = element[0][index]
            print(img.shape, img.max(), element[1][index])


if __name__ == "__main__":
    main()
