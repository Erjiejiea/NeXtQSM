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

from network import varnet
from network.unet import UNet
from utils.data_loader import get_loader
from utils.data_loader_nii import gen_dataset, gen_dataset_test
from utils.misc import mkexperiment, get_act_function, plot_history
from utils.save_image import SaveImageCallback


def main(config):
    # check gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # print(tf.config.experimental.list_physical_devices('GPU'))
    # tf.debugging.set_log_device_placement(True)

    # random seed
    np.random.seed(2)

    # set up experiment
    experiment_path = mkexperiment(config, cover=True)
    inter_result_path = os.path.join(experiment_path, 'inter_result')
    model_path = os.path.join(config.model_path, config.name)
    bf_checkpoint_path = model_path + '/' + config.name + '_epoch_{epoch}.ckpt'

    # load data
    train_dataset = get_loader(config.train_input_path, config.train_gt_path, config,
                               config.BATCH_SIZE, config.crop_key, mode='train')
    val_dataset = get_loader(config.val_input_path, config.val_gt_path, config,
                               config.BATCH_SIZE, config.crop_key, mode='train')
    test_dataset = get_loader(config.test_input_path, config.test_gt_path, config,
                              1, True, mode='brain')

    # print('Loaded {} samples for training.'.format(len(train_x_dataset)))
    # print('Loaded {} samples for validation.'.format(len(val_x_dataset)))
    # print('Loded {} real brain.'.format(len(test_x_dataset)))

    # print('Loded {} real brain.'.format(tf.data.experimental.cardinality(test_x_dataset).numpy()))
    # print('Loaded {} samples for training.'.format(
    #     tf.data.experimental.cardinality(train_dataset).numpy() * config.BATCH_SIZE))
    # print('Loaded {} samples for validation.'.format(
    #     tf.data.experimental.cardinality(val_dataset).numpy() * config.BATCH_SIZE))

    # model: VN
    vn_network = varnet.VarNet(config)
    vn_network.summary((256, 256, 256, 1),)

    # cost function & optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.09, beta_2=0.009)

    # train
    # create checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=bf_checkpoint_path,
                                                    save_freq="epoch",
                                                    # save_best_only=True,
                                                    period=config.save_period)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    cp_callpack = SaveImageCallback(save_dir_inter_result=inter_result_path,
                                    interval=config.save_period,
                                    real_data=test_dataset,
                                    crange=[-0.1, 0.1])

    # with tf.device('/GPU:0'):
    vn_network.compile(loss=loss_fn, optimizer=optimizer)  #metrics=['accuracy', 'val_loss']
    print('# Fit bf_model on training data')
    vn_history = vn_network.fit(train_dataset,
                                epochs=config.epochs_train,
                                callbacks=[checkpoint, cp_callpack],
                                shuffle=True,
                                validation_data=val_dataset,
                                verbose=1)
    # pass callback to training for saving the model

    loss_vn_history = vn_history.history['loss']
    print('Loss: ', loss_vn_history)

    plot_history(vn_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment info
    parser.add_argument('--name', type=str, default='version2')
    parser.add_argument('--experiment_path', type=str, default='')
    parser.add_argument('--train_input_path', type=str, default='./data_val/train_localfield/')
    parser.add_argument('--train_gt_path', type=str, default='./data_val/train_chimap/')
    parser.add_argument('--val_input_path', type=str, default='./data_val/val_localfield/')
    parser.add_argument('--val_gt_path', type=str, default='./data_val/val_chimap/')
    parser.add_argument('--test_input_path', type=str, default='./data_val/real_localfield/')
    parser.add_argument('--test_gt_path', type=str, default='./data_val/real_chimap/')
    parser.add_argument('--GPU_NUM', type=str, default='1')   # 3[0], 4[2], 5[4], 6[5], 7[6]

    # model hyper-parameters
    # parser.add_argument('--OUTPUT_C', type=int, default=1)  # OUTPUT CHANNELS
    parser.add_argument('--n_layers', type=int, default=7)
    parser.add_argument('--starting_filters', type=int, default=16)
    parser.add_argument('--kernel_initializer', type=str, default='he_normal')  # he_normal
    parser.add_argument('--batch_norm', type=bool, default=False)

    parser.add_argument('--act_func', type=str, default='relu')
    parser.add_argument('--conv_per_layer', type=int, default=1)

    # training hyper-parameters
    parser.add_argument('--all_batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0003)  # .0003

    parser.add_argument('--BATCH_SIZE', type=int, default=2)
    parser.add_argument('--INPUT_H', type=int, default=256)
    parser.add_argument('--INPUT_W', type=int, default=256)
    parser.add_argument('--INPUT_D', type=int, default=256)
    parser.add_argument('--crop_key', type=bool, default=True)
    parser.add_argument('--CROP_SIZE', type=int, default=64)

    parser.add_argument('--weight_vn', type=float, default=10.0)
    parser.add_argument('--vn_n_steps', type=int, default=6)
    parser.add_argument('--vn_batch_norm', type=bool, default=False)
    parser.add_argument('--vn_batch_size', type=int, default=2)
    parser.add_argument('--vn_lr', type=float, default=0.005)
    parser.add_argument('--vn_n_layers', type=int, default=5)
    parser.add_argument('--vn_kernel_initializer', type=str, default='he_normal')
    parser.add_argument('--vn_act_func', type=str, default='relu')
    parser.add_argument('--vn_l_init', type=float, default=0.1)
    parser.add_argument('--vn_starting_filters', type=int, default=16)
    parser.add_argument('--vn_dt_loss', type=str, default='RMSE')

    parser.add_argument('--epochs_train', type=int, default=100)
    parser.add_argument('--save_period', type=int, default=1)

    # misc
    parser.add_argument('--model_path', type=str, default='./models/vn/')  # phase unwrapped (totalfield) to phase tissue (localfield)
    parser.add_argument('--result_path', type=str, default='./results/vn/')

    config = parser.parse_args()
    main(config)
