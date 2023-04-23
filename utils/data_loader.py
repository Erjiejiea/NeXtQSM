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

    def __init__(self, input_root, gt_root, config, crop_key, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.mode = mode
        self.input_root = input_root
        self.gt_root = gt_root
        self.batch_size = config.BATCH_SIZE

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
        # batch_gt_paths = self.gt_paths[start:end]

        if self.mode == 'brain':
            batch_input = np.array([self.get_data_pair(i) for i in batch_input_paths])
            return batch_input
        else:
            batch_input, batch_gt = np.array([self.get_data_pair(i) for i in batch_input_paths])
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
        image_path = self.image_paths[index]
        gt_path = self.gt_paths[index]
        if self.mode == 'brain':
            image = load_data_nii_test(image_path)
        else:
            image, gt = load_data_nii(image_path, gt_path)

        if self.crop_key:
            # -----RandomCrop----- #
            (h, w, d) = image.shape
            th, tw, td = self.crop_size, self.crop_size, self.crop_size
            drop_voxel = self.config.INPUT_H * 0.3
            i = random.randint(drop_voxel, h - th - drop_voxel)
            j = random.randint(drop_voxel, w - tw - drop_voxel)
            k = random.randint(drop_voxel, d - td - drop_voxel)
            if h <= th and w <= tw and d <= td:
                print('Error! Your input size is too small: %d is smaller than crop size %d ' % (w, self.crop_size))
                return
            image = image[i:i + th, j:j + tw, k:k + td]
            gt = gt[i:i + th, j:j + tw, k:k + td]

        if self.mode == 'brain':
            return image
        else:
            return image, gt


def get_loader(input_path, gt_path, config, crop_key, shuffle=True, mode='train'):
    """Builds and returns Dataloader."""

    data_loader = ImageFolder(input_root=input_path, gt_root=gt_path, config=config, crop_key=crop_key, mode=mode)

    return data_loader
