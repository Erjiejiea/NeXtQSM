# -*- coding: utf-8 -*-
'''

 @Time     : 3/28/23, 2:55 PM
 @Author   : yqq
 @Modified : Jie, 4/18/23, 7:52 PM

'''

import argparse
import csv
import os
import numpy as np
import tensorflow as tf
import nibabel as nib

from network.unet import UNet
from utils.misc import get_act_function
from network import varnet


def main(config):
    # check gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # random seed
    np.random.seed(2)

    # set up experiment
    bf_checkpoint_path = os.path.join(config.model_path_bf,
                                      config.name_bf) + '/' + config.name_bf + '_epoch_' + config.model_num_bf + '.ckpt'
    vn_checkpoint_path = os.path.join(config.model_path_vn,
                                      config.name_vn) + '/' + config.name_vn + '_epoch_' + config.model_num_vn + '.ckpt'

    # model: BF
    bf_network = UNet(1, config.n_layers, config.starting_filters, 3, config.kernel_initializer, config.batch_norm,
                      0., get_act_function(config.act_func), config.conv_per_layer, False, False, None)
    bf_network.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.09,
                                                          beta_2=0.009))
    bf_network.load_weights(bf_checkpoint_path)

    # model: VN
    vn_network = varnet.VarNet(config)
    vn_network.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.09, beta_2=0.009))
    vn_network.load_weights(vn_checkpoint_path)

    for filename in os.listdir(config.bf_input_path):
        name = filename.split('_')[1]
        bf_input = nib.load(config.bf_input_path + '/totalfield_' + name).get_fdata()
        # bf_gt = nib.load(config.bf_gt_path + '/localfield_' + name).get_fdata()  # localfield
        # vn_gt = nib.load(config.vn_gt_path + '/image_' + name).get_fdata()  # chimap

        # expand dims
        bf_input = np.expand_dims(bf_input, -1)
        bf_input = np.expand_dims(bf_input, 0)

        predict_bf = bf_network.predict(bf_input)
        predict_vn = vn_network.predict(predict_bf)

        break

    # save whole model
    save_dir_bf = config.model_path_bf + '_wholemodel/' + config.name_bf + '/' + config.name_bf + '_epoch_' + config.model_num_bf
    bf_network.save(save_dir_bf)

    save_dir_vn = config.model_path_vn + '_wholemodel/' + config.name_vn + '/' + config.name_vn + '_epoch_' + config.model_num_vn
    vn_network.save(save_dir_vn)
    print('Successfully save vn_network: ', save_dir_vn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment info
    parser.add_argument('--bf_input_path', type=str, default='./data/real_totalfield/')
    parser.add_argument('--bf_gt_path', type=str, default='./data/real_localfield/')
    parser.add_argument('--vn_gt_path', type=str, default='./data/real_chimap/')
    parser.add_argument('--GPU_NUM', type=str, default='7')  # 3[0], 4[2], 5[4], 6[5], 7[6]

    # BF
    # model hyper-parameters
    parser.add_argument('--n_layers', type=int, default=7)
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

    # VN
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

    # misc
    parser.add_argument('--name_bf', type=str, default='v2_7layer')
    parser.add_argument('--name_vn', type=str, default='v2_7layer')
    parser.add_argument('--model_path_bf', type=str, default='./models/bf')  # total_field to local_field
    parser.add_argument('--model_path_vn', type=str, default='./models/vn')
    parser.add_argument('--model_num_bf', type=str, default='12')
    parser.add_argument('--model_num_vn', type=str, default='2')

    config = parser.parse_args()
    main(config)
