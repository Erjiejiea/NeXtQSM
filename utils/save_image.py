# -*- coding: utf-8 -*-
'''

 @Time   : 4/20/23, 11:12 AM
 @Author : Jie
 
'''

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.misc import save_numpy_result


class SaveImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir_inter_result, interval, real_data, crange):
        super(SaveImageCallback, self).__init__()
        self.save_dir_inter_result = save_dir_inter_result
        self.interval = interval
        self.real_data = real_data
        self.crange = crange

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            real_data_pred = self.model.predict(self.real_data)

            # save pictures
            picture_file_path = os.path.join(self.save_dir_inter_result, "real_epoch{}".format(epoch + 1))
            save_numpy_result(real_data_pred[:, :, :, 120, :], save_dir=picture_file_path,
                              format='png', cmap='gray', norm=False, crange=self.crange)

            # free the memory
            # del real_data_pred


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_num):
        super(MyCallback, self).__init__()
        self.epoch_num = epoch_num

    def on_epoch_end(self, epoch, logs=None):
        # print(f"Epoch {epoch+1}/{self.epoch_num}  ")
        print(f"Training loss: {logs['loss']:.4f}, Validation loss: {logs['val_loss']}")
