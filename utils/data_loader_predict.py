import numpy as np
import nibabel as nib
import tensorflow as tf
from keras.backend import cast_to_floatx

from utils import misc

import os
import random
import math
import numpy as np
import nibabel as nib

from tensorflow.keras.utils import Sequence


def load_data_nii(image_path, gt_path):
    input_sample = nib.load(image_path).get_fdata()
    gt_sample = nib.load(gt_path).get_fdata()

    return input_sample, gt_sample


def load_data_nii_test(image_path):
    input_sample = nib.load(image_path).get_fdata()

    return input_sample


class ImageFolder(Sequence):
    """Load Variaty Chinese Fonts for Iterator. """

    def __init__(self, input_root, gt_root, config, batch_size, crop_key=False, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.mode = mode
        self.input_root = input_root
        self.gt_root = gt_root
        self.batch_size = batch_size

        self.crop_key = crop_key
        self.crop_size = config.CROP_SIZE

        self.image_paths = list(map(lambda x: os.path.join(self.input_root, x), os.listdir(self.input_root)))
        self.gt_paths = list(map(lambda x: os.path.join(self.gt_root, x), os.listdir(self.gt_root)))
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))
        print("label count in {} path :{}".format(self.mode, len(self.gt_paths)))
        self.image_paths.sort(reverse=True)
        self.gt_paths.sort(reverse=True)

    def __getitem__(self, index):
        """Reads a image batch from a file batch and preprocesses them and returns."""
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.image_paths))
        batch_input_paths = self.image_paths[start:end]
        batch_gt_paths = self.gt_paths[start:end]

        if self.mode == 'brain':
            for i in range(len(batch_input_paths)):
                input_array, gt_array = self.get_data_pair(i)
                if i == 0:
                    batch_input = input_array
                else:
                    batch_input = np.concatenate((batch_input, input_array), axis=0)

            # print('batch_input shape for test: ', batch_input.shape, ' Index: ', index)
            return batch_input

        else:
            for i in range(len(batch_input_paths)):

                input_array, gt_array = self.get_data_pair(i)
                if i == 0:
                    batch_input = input_array
                    batch_gt = gt_array
                else:
                    batch_input = np.append(batch_input, input_array, axis=0)
                    batch_gt = np.append(batch_gt, gt_array, axis=0)

            # print('batch_input shape for train: ', batch_input.shape, ' Index: ', index)

            return batch_input, batch_gt

        # -----To Tensor------#
        # batch_input = tf.convert_to_tensor(batch_input)
        # batch_gt = tf.convert_to_tensor(batch_gt)
        # batch_input = cast_to_floatx(batch_input)
        # batch_gt = cast_to_floatx(batch_gt)

    def __len__(self):
        """Returns the total number of font files in one BATCH_SIZE."""
        return math.ceil(len(self.image_paths) / self.batch_size)

    def get_data_pair(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        if self.mode == 'brain':
            # image_path = self.image_paths[index]
            # image = load_data_nii_test(image_path)
            #
            # image = np.expand_dims(image, -1)
            # image = np.expand_dims(image, 0)

            image_path = self.image_paths[index]
            gt_path = self.gt_paths[index]
            image, gt = load_data_nii(image_path, gt_path)
            image = np.expand_dims(image, -1)
            image = np.expand_dims(image, 0)
            gt = np.expand_dims(gt, -1)
            gt = np.expand_dims(gt, 0)

            return image, gt

        else:
            image_path = self.image_paths[index]
            gt_path = self.gt_paths[index]
            image, gt = load_data_nii(image_path, gt_path)

            if self.crop_key:
                # -----RandomCrop----- #
                (h, w, d) = image.shape
                th, tw, td = self.crop_size, self.crop_size, self.crop_size
                drop_voxel = np.ceil(self.config.INPUT_H * 0.3)
                i = random.randint(drop_voxel, h - th - drop_voxel)
                j = random.randint(drop_voxel, w - tw - drop_voxel)
                k = random.randint(drop_voxel, d - td - drop_voxel)
                if h <= th and w <= tw and d <= td:
                    print('Error! Your input size is too small: %d is smaller than crop size %d ' % (w, self.crop_size))
                    return
                image = image[i:i + th, j:j + tw, k:k + td]
                gt = gt[i:i + th, j:j + tw, k:k + td]

            image = np.expand_dims(image, -1)
            image = np.expand_dims(image, 0)
            gt = np.expand_dims(gt, -1)
            gt = np.expand_dims(gt, 0)

            return image, gt


def get_loader_predict(input_path, gt_path, config, batch_size, crop_key, mode='real'):
    """Builds and returns Dataloader."""

    data_loader = ImageFolder(input_root=input_path, gt_root=gt_path, config=config,
                              batch_size=batch_size, crop_key=crop_key, mode=mode)

    return data_loader
