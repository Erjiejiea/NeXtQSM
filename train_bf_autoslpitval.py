# -*- coding: utf-8 -*-
'''

 @Time     : 3/28/23, 2:55 PM
 @Author   : yqq
 @Modified : Jie, 4/18/23, 7:52 PM

'''

import argparse
import os
import numpy as np
import tensorflow as tf

from network.unet import UNet
from utils.data_loader import get_loader
from utils.misc import mkexperiment, get_act_function
from utils.save_image import SaveImageCallback, MyCallback


def main(config):
    # check gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    # print(tf.config.experimental.list_physical_devices('GPU'))
    # tf.debugging.set_log_device_placement(True)

    # random seed
    np.random.seed(2)

    # set up experiment
    experiment_path = mkexperiment(config, cover=True)
    inter_result_path = os.path.join(experiment_path, 'inter_result')
    model_path = os.path.join(config.model_path, config.name)
    bf_checkpoint_path = model_path + '_bf_{epoch}.ckpt'

    # load data
    train_dataset = get_loader(config.train_input_path, config.train_gt_path,
                               config, config.crop_key, shuffle=True, mode='train')
    test_dataset = get_loader(config.test_input_path, config.test_gt_path,
                              config, crop_key=False, shuffle=True, mode='brain')

    # model: BF
    bf_network = UNet(1, config.n_layers, config.starting_filters, 3, config.kernel_initializer, config.batch_norm,
                      0., get_act_function(config.act_func), config.conv_per_layer, False, False, None)
    bf_network.summary((256, 256, 256, 1),)  # (64, 64, 64, 1)

    # cost function & optimizer
    bf_network.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.09, beta_2=0.009),
                       metrics=[tf.keras.metrics.MeanSquaredError()])

    # train
    # create checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=bf_checkpoint_path,
                                                    save_freq="epoch",
                                                    save_best_only=True,
                                                    verbose=1,
                                                    period=config.save_period)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    cp_callback = SaveImageCallback(save_dir_inter_result=inter_result_path,
                                    interval=config.save_period,
                                    real_data=test_dataset,
                                    crange=[-0.1, 0.1])
    # log_callback = MyCallback(epoch_num=config.epochs_train)

    # strategy = tf.distribute.OneDeviceStrategy(device='/gpu:'+config.GPU_NUM)
    # with strategy.scope():
    print('# Fit bf_model on training data')
    bf_history = bf_network.fit(train_dataset,
                                epochs=config.epochs_train,
                                callbacks=[checkpoint, cp_callback],
                                shuffle=True,
                                validation_split=0.1,
                                verbose=2)  # pass callback to training for saving the model

    loss_bf_history = bf_history.history['loss']
    print('Loss: ', loss_bf_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment info
    parser.add_argument('--name', type=str, default='version1')
    parser.add_argument('--experiment_path', type=str, default='')
    parser.add_argument('--train_input_path', type=str, default='./data/train_totalfield/')
    parser.add_argument('--train_gt_path', type=str, default='./data/train_localfield/')
    parser.add_argument('--test_input_path', type=str, default='./data/real_totalfield/')
    parser.add_argument('--test_gt_path', type=str, default='./data/real_localfield/')
    parser.add_argument('--GPU_NUM', type=str, default='4')  # 3[0], 4[2], 5[4], 6[5], 7[6]

    # dataset parameters
    parser.add_argument('--BATCH_SIZE', type=int, default=1)
    parser.add_argument('--INPUT_H', type=int, default=256)
    parser.add_argument('--INPUT_W', type=int, default=256)
    parser.add_argument('--INPUT_D', type=int, default=256)
    parser.add_argument('--crop_key', type=bool, default=False)
    parser.add_argument('--CROP_SIZE', type=int, default=64)

    # model hyper-parameters
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--starting_filters', type=int, default=16)
    parser.add_argument('--kernel_initializer', type=str, default='he_normal')  # he_normal
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--act_func', type=str, default='relu')
    parser.add_argument('--conv_per_layer', type=int, default=1)

    # training hyper-parameters
    parser.add_argument('--all_batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=4e-4)  # .0003
    parser.add_argument('--epochs_train', type=int, default=100)
    parser.add_argument('--save_period', type=int, default=1)

    # misc
    parser.add_argument('--model_path', type=str, default='./models/')  # total_field to local_field
    parser.add_argument('--result_path', type=str, default='./results/bf/')

    config = parser.parse_args()
    main(config)
