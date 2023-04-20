# -*- coding: utf-8 -*-
'''

 @Time   : 4/20/23, 11:12 AM
 @Author : Jie
 
'''

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class SaveImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, interval, real_data):
        super(SaveImageCallback, self).__init__()
        self.save_dir = save_dir
        self.interval = interval
        self.real_data = real_data

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            # save model
            model_path = os.path.join(self.save_dir)
            self.model.save(model_path)

            # save val data results
            _,_,x_test, y_test = self.model.validation_data
            y_pred = self.model.predict(x_test)
            fig, axes = plt.subplots(nrows=5,ncols=5, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(y_pred[i].reshape((28, 28), cmap='gray'))  # x_test[i]?
                ax.set_xtricks([])
                ax.set_ytricks([])
                ax.set_xlabel("pred: {}".format(np.argmax(y_pred[i])))
                ax.set_ylabel("true: {}".format(y_test[i]))
            fig.canvas.draw()
            png_data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close()
            png_encoded = tf.image.encode_png(png_data)
            png_file_path = os.path.join(self.save_dir, "pred_val_epoch{}.png".format(epoch + 1))
            tf.io.write_file(png_file_path, png_encoded)

            # save test data results (real data)
            real_data_pred = self.model.predict(self.real_data)
            fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(real_data_pred[i].reshape((28, 28), cmap='gray'))
                ax.set_xtricks([])
                ax.set_ytricks([])
                ax.set_xlabel([])
                ax.set_ylabel([])
            fig.canvas.draw()
            png_data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close()
            png_encoded = tf.image.encode_png(png_data)
            png_file_path = os.path.join(self.save_dir, "pred_real_epoch{}.png".format(epoch + 1))
            tf.io.write_file(png_file_path, png_encoded)