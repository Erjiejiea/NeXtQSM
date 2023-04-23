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


def gen_data_tuples(train_dir, label_dir, validation_split=0.2):
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
def generator_function(data_tuples, batch_size):
    def generator():
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i+batch_size]
            x_batch = [nib.load(file_path).get_fdata() for file_path, _ in batch]
            y_batch = [nib.load(file_path).get_fdata() for _, file_path in batch]
            x_batch = np.expand_dims(x_batch, axis=-1)
            y_batch = np.expand_dims(y_batch, axis=-1)
            yield tf.stack(x_batch), tf.stack(y_batch)
    return generator


def generator_function_test(data_tuples, batch_size):
    def generator():
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i+batch_size]
            x_batch = [nib.load(file_path[0]).get_fdata() for file_path in batch]
            x_batch = np.expand_dims(x_batch, axis=-1)
            yield tf.stack(x_batch)
    return generator


def gen_dataset(train_dir, label_dir, validation_split, batch_size):
    train_tuples, val_tuples = gen_data_tuples(train_dir, label_dir, validation_split=validation_split)

    train_generator = generator_function(train_tuples, batch_size)
    val_generator = generator_function(val_tuples, batch_size)

    train_dataset = tf.data.Dataset.from_generator(
        train_generator, output_types=(tf.float32, tf.float32),
        output_shapes=((batch_size, None, None, None, 1), (batch_size, None, None, None, 1)))

    val_dataset = tf.data.Dataset.from_generator(
        val_generator, output_types=(tf.float32, tf.float32),
        output_shapes=((batch_size, None, None, None, 1), (batch_size, None, None, None, 1)))

    train_x_dataset = train_dataset.take(1)
    train_y_dataset = train_dataset.take(2)
    val_x_dataset = val_dataset.take(1)
    val_y_dataset = val_dataset.take(2)

    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_x_dataset, train_y_dataset, val_x_dataset, val_y_dataset


def gen_dataset_test(train_dir, label_dir):
    # obtain file list
    train_files = sorted(os.listdir(train_dir))
    label_files = sorted(os.listdir(label_dir))

    train_paths = [os.path.join(train_dir, f) for f in train_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]
    x_tuples = list(zip(train_paths))
    y_tuples = list(zip(label_paths))

    batch_size = 1
    # test_data_gen = data_generator(data_tuples, batch_size)
    # test_dataset = tf.data.Dataset.from_generator(
    #     test_data_gen, (tf.float32, tf.float32),
    #     (tf.TensorShape([batch_size, None, None, None]), tf.TensorShape([batch_size, None, None, None]))
    # )

    test_x_generator = generator_function_test(x_tuples, batch_size)
    test_y_generator = generator_function_test(y_tuples, batch_size)
    x_dataset = tf.data.Dataset.from_generator(test_x_generator,
                                               output_types=tf.float32,
                                               output_shapes=(batch_size, None, None, None, 1))
    y_dataset = tf.data.Dataset.from_generator(test_y_generator,
                                               output_types=tf.float32,
                                               output_shapes=(batch_size, None, None, None, 1))

    # x_dataset = x_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # y_dataset = y_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return x_dataset, y_dataset
