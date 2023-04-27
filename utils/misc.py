import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import math

import os
import math
import shutil
import matplotlib.pyplot as plt
import pandas as pd


def get_base_path(prefix=""):
    base_path = str(Path(__file__).parent.parent.parent) + "/"
    return base_path


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def add_padding(volumes, pad_size):
    assert (len(volumes.shape) == 5 and len(pad_size) == 3)
    padded_volumes = []
    shape = volumes.shape[1:]
    for volume in volumes:
        # Add one if shape is not EVEN
        padded = np.pad(volume[:, :, :, 0], [(int(shape[0] % 2 != 0), 0), (int(shape[1] % 2 != 0), 0), (int(shape[2] % 2 != 0), 0)], 'constant', constant_values=0.0)

        # Calculate how much padding to give
        val_x = (pad_size[0] - padded.shape[0]) // 2
        val_y = (pad_size[1] - padded.shape[1]) // 2
        val_z = (pad_size[2] - padded.shape[2]) // 2

        # Append padded volume
        padded_volumes.append(np.pad(padded, [(val_x,), (val_y,), (val_z,)], 'constant', constant_values=0.0))

    padded_volumes = np.array(padded_volumes)
    assert (padded_volumes.shape[1] == pad_size[0] and padded_volumes.shape[2] == pad_size[1] and padded_volumes.shape[3] == pad_size[2])

    return np.expand_dims(padded_volumes, -1), np.array(shape[:-1]), np.array([val_x, val_y, val_z])


def remove_padding(volumes, orig_shape, values):
    assert (len(volumes.shape) == 5 and len(orig_shape) == 3 and len(values) == 3)
    # Remove padding
    if values[0] != 0:
        volumes = volumes[:, values[0]:-values[0], :, :]
    if values[1] != 0:
        volumes = volumes[:, :, values[1]:-values[1], :]
    if values[2] != 0:
        volumes = volumes[:, :, :, values[2]:-values[2]]

    volumes = volumes[:, int(orig_shape[0] % 2 != 0):, int(orig_shape[1] % 2 != 0):, int(orig_shape[2] % 2 != 0):]
    assert (volumes.shape[1] == orig_shape[0] and volumes.shape[2] == orig_shape[1] and volumes.shape[3] == orig_shape[2])

    return volumes


def get_how_much_to_pad(shape, multiple):
    pad = []
    is_same_shape = True
    for val in shape:
        pad.append(math.ceil(val/multiple) * multiple)
        if pad[-1] != val:
            is_same_shape = False

    return pad, is_same_shape


def get_act_function(label):
    return tf.keras.layers.ReLU


####  above: from NeXtQSM-main misc
####  following: from from yqq-master misc

def save_tf_result_multi(tensor, save_dir,
                      nrow=1, padding=0, pad_value=0,
                      format='jpg', cmap='gray', norm=False, crange=[0, 1]):
    """Save a given Tensor into an image file.
    """
    nmaps = tensor.shape[0]  # num of channels

    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = tf.fill((tensor.shape[1], height * ymaps + padding, width * xmaps + padding),pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height - padding,
            x * width + padding: (x + 1) * width - padding].assign(tensor[k])
            k = k + 1
    merge_img = np.squeeze(grid.numpy().transpose(1, 2, 0))
    if norm:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap)
    else:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap, vmin=crange[0], vmax=crange[1])


def save_tf_result(tensor, save_dir,
                      nrow=4, padding=0, pad_value=0,
                      format='jpg', cmap='gray', norm=False, crange=[0, 1]):
    """Save a given Tensor into an image file.
    """
    nmaps = tensor.shape[0]

    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = tf.fill((tensor.shape[1], height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height - padding,
            x * width + padding: (x + 1) * width - padding].assign(tensor[k])
            k = k + 1
    merge_img = np.squeeze(grid.numpy().transpose(1, 2, 0))
    if norm:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap)
    else:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap, vmin=crange[0], vmax=crange[1])


def save_numpy_result(nparray, save_dir,
                      nrow=4, padding=0, pad_value=0,
                      format='jpg', cmap='gray', norm=False, crange=[0, 1]):
    """Save a given numpy into an image file.
    """
    nmaps = nparray.shape[0]

    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(nparray.shape[1] + padding), int(nparray.shape[2] + padding)
    grid = np.full((nparray.shape[3], height * ymaps + padding, width * xmaps + padding), pad_value, dtype='float32')
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            a = nparray[k, :, :, :]
            b = np.squeeze(nparray[k, :, :, :])
            grid[:, y * height + padding:(y + 1) * height - padding,
            x * width + padding: (x + 1) * width - padding] = np.squeeze(nparray[k, :, :, :])
            k = k + 1
    merge_img = np.squeeze(np.transpose(grid, axes=[1, 2, 0]))
    if norm:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap)
    else:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap, vmin=crange[0], vmax=crange[1])


def save_torch_result_with_label(tensor, label, save_dir, loss=False,
                                 nrow=4, padding=0, pad_value=0,
                                 format='jpg', cmap='gray', norm=False, crange=[0, 1]):
    nmaps = tensor.shape[0]
    if loss:
        tensor = tf.concat([tensor, label, tf.abs(tensor-label)], axis=3)
    else:
        tensor = tf.concat([tensor, label], axis=3)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = tf.fill((tensor.shape[1], height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y+1) * height - padding,
            x * width + padding : (x + 1) * width - padding].assign(tensor[k])
            k = k + 1
    merge_img = np.squeeze(tf.transpose(grid, perm=[1, 2, 0]).numpy())

    if norm:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap)
    else:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap, vmin=crange[0], vmax=crange[1])


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def mkexperiment(config,cover=False):
    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    experiment_path = os.path.join(config.result_path,config.name)
    if os.path.exists(experiment_path):
        if cover:
            shutil.rmtree(experiment_path)
            os.makedirs(experiment_path)
            # os.makedirs(os.path.join(experiment_path, 'tensorboard'))
            os.makedirs(os.path.join(experiment_path, 'inter_result'))
        else:
            raise ValueError("Experiment '{}' already exists. Please modify the experiment name!"
                             .format(config.name))
    else:
        os.makedirs(experiment_path)
        # os.makedirs(os.path.join(experiment_path, 'tensorboard'))
        os.makedirs(os.path.join(experiment_path, 'inter_result'))
    return experiment_path


def plot_history(history):
    hist = pd.DataFrame(history.histpry)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.show()

