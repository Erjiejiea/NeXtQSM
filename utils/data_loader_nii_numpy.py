# -*- coding: utf-8 -*-
'''

 @Time   : 4/20/23, 8:08 PM
 @Author : Jie
 
'''

import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def gen_data_tuples(train_dir, label_dir, validation_split=0.1):
    # obtain file list
    train_files = sorted(os.listdir(train_dir))
    label_files = sorted(os.listdir(label_dir))

    train_paths = [os.path.join(train_dir, f) for f in train_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]
    data_tuples = list(zip(train_paths, label_paths))

    # split to training set and val set
    train_tuples, val_tuples = train_test_split(data_tuples, test_size=validation_split)

    return train_tuples, val_tuples


# def data_generator(data_tuples, bacth_size):
#     while True:
#         np.random.shuffle(data_tuples)
#         for start in range(0, len(data_tuples), bacth_size):
#             end = min(start + bacth_size, len(data_tuples))
#
#             batch_train_data = []
#             batch_label_data = []
#             for train_data_path, label_data_path in data_tuples[start:end]:
#                 train_data = nib.load(train_data_path).get_fdata()
#                 label_data = nib.load(label_data_path).get_fdata()
#
#                 batch_train_data.append(train_data)
#                 batch_label_data.append(label_data)
#             yield np.array(batch_train_data), np.array(batch_label_data)
def generate_function(data_tuples, batch_size):
    for i in range(0, len(data_tuples), batch_size):
        batch = data_tuples[i:i+batch_size]
        x_batch = [nib.load(file_path).get_fdata() for file_path, _ in batch]
        y_batch = [nib.load(file_path).get_fdata() for _, file_path in batch]
        x_batch = np.expand_dims(x_batch, axis=-1)
        y_batch = np.expand_dims(y_batch, axis=-1)
    return x_batch, y_batch


def gen_dataset_numpy(train_dir, label_dir, validation_split, batch_size):
    train_tuples, val_tuples = gen_data_tuples(train_dir, label_dir, validation_split=validation_split)

    train_x_gen, train_y_gen = generate_function(train_tuples, batch_size)
    val_x_gen, val_y_gen = generate_function(val_tuples, batch_size)

    train_x_tensor = tf.convert_to_tensor(train_x_gen)
    train_y_tensor = tf.convert_to_tensor(train_y_gen)
    val_x_tensor = tf.convert_to_tensor(val_x_gen)
    val_y_tensor = tf.convert_to_tensor(val_y_gen)

    return train_x_tensor, train_y_tensor, val_x_tensor, val_y_tensor


def gen_dataset_test(train_dir, label_dir):
    # obtain file list
    train_files = sorted(os.listdir(train_dir))
    label_files = sorted(os.listdir(label_dir))

    train_paths = [os.path.join(train_dir, f) for f in train_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]
    data_tuples = list(zip(train_paths, label_paths))

    batch_size = 1

    test_x_gen, test_y_gen = generate_function(data_tuples, batch_size)
    test_x_tensor = tf.convert_to_tensor(test_x_gen)
    test_y_tensor = tf.convert_to_tensor(test_y_gen)

    return test_x_tensor, test_y_tensor
