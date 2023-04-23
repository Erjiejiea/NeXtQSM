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
from utils.data_loader_nii import gen_dataset, gen_dataset_test
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
    train_dataset, val_dataset = gen_dataset(config.totalfield_path,
                                             config.localfield_path,
                                             validation_split=0.1,
                                             batch_size=config.BATCH_SIZE)
    test_x_dataset, test_y_dataset = gen_dataset_test(os.path.join(config.realdata_path, 'totalfield'),
                                    os.path.join(config.realdata_path, 'localfield'))

    # print('Loaded {} samples for training.'.format(len(train_x_dataset)))
    # print('Loaded {} samples for validation.'.format(len(val_x_dataset)))
    # print('Loded {} real brain.'.format(len(test_x_dataset)))

    # print('Loded {} real brain.'.format(tf.data.experimental.cardinality(test_x_dataset).numpy()))
    # print('Loaded {} samples for training.'.format(
    #     tf.data.experimental.cardinality(train_dataset).numpy() * config.BATCH_SIZE))
    # print('Loaded {} samples for validation.'.format(
    #     tf.data.experimental.cardinality(val_dataset).numpy() * config.BATCH_SIZE))

    # model: BF
    bf_network = UNet(1, config.n_layers, config.starting_filters, 3, config.kernel_initializer, config.batch_norm,
                      0., get_act_function(config.act_func), config.conv_per_layer, False, False, None)
    bf_network.summary((256, 256, 256, 1),)  # (64, 64, 64, 1)

    # cost function & optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.09, beta_2=0.009)

    # train
    # create checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=bf_checkpoint_path,
                                                    save_freq="epoch",
                                                    save_best_only=True,
                                                    period=config.save_period)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    cp_callback = SaveImageCallback(save_dir_inter_result=inter_result_path,
                                    interval=config.save_period,
                                    real_data=[test_x_dataset, test_y_dataset],
                                    crange=[-0.1, 0.1])
    # log_callback = MyCallback(epoch_num=config.epochs_train)

    # strategy = tf.distribute.OneDeviceStrategy(device='/gpu:'+config.GPU_NUM)
    # with strategy.scope():
    bf_network.compile(loss=loss_fn, optimizer=optimizer)  # metrics=['accuracy', 'val_loss']
    print('# Fit bf_model on training data')
    bf_history = bf_network.fit(train_dataset,
                                # batch_size=config.BATCH_SIZE,
                                epochs=config.epochs_train,
                                callbacks=[checkpoint, cp_callback],
                                shuffle=True,
                                validation_data=val_dataset,
                                verbose=2)
        # pass callback to training for saving the model

    loss_bf_history = bf_history.history['loss']
    print('Loss: ', loss_bf_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment info
    parser.add_argument('--name', type=str, default='version1')
    parser.add_argument('--experiment_path', type=str, default='')
    parser.add_argument('--sus_path', type=str, default='/DATA_Inter2/cj/NeXtQSM/train/train_synthetic_brain/')
    parser.add_argument('--localfield_path', type=str, default='/DATA_Inter2/cj/NeXtQSM/train_localfield/')
    parser.add_argument('--totalfield_path', type=str, default='/DATA_Inter2/cj/NeXtQSM/train_totalfield/')
    parser.add_argument('--realdata_path', type=str, default='/DATA_Inter2/cj/NeXtQSM/realdata_for_NeXtQSM/')
    parser.add_argument('--GPU_NUM', type=str, default='4')  # 3[0], 4[2], 5[4], 6[5], 7[6]

    # model hyper-parameters
    # parser.add_argument('--OUTPUT_C', type=int, default=1)  # OUTPUT CHANNELS
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--starting_filters', type=int, default=16)
    parser.add_argument('--kernel_initializer', type=str, default='he_normal')  # he_normal
    parser.add_argument('--batch_norm', type=bool, default=False)

    parser.add_argument('--act_func', type=str, default='relu')
    parser.add_argument('--conv_per_layer', type=int, default=1)

    # training hyper-parameters
    parser.add_argument('--all_batch_size', type=int, default=1)
    parser.add_argument('--BATCH_SIZE', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=4e-4)  # .0003

    parser.add_argument('--epochs_train', type=int, default=100)
    parser.add_argument('--save_period', type=int, default=1)

    # misc
    parser.add_argument('--model_path', type=str, default='./models/')  # phase unwrapped (totalfield) to phase tissue (localfield)
    parser.add_argument('--result_path', type=str, default='./results/bf/')

    config = parser.parse_args()
    main(config)