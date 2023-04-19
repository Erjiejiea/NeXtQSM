# -*- coding: utf-8 -*-
'''

 @Time     :
 @Author   : yqq
 @Modified : Jie, 3/28/23, 2:55 PM

'''

import argparse
import os
import numpy as np
import tensorflow as tf

from network.unet import UNet
from network.ops import net_analysis
from utils.data_loader import load_training_volume
from utils.misc import mkexperiment, get_base_path, get_act_function


def main(config):
    # check gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM

    # random seed
    np.random.seed(2)

    # set up experiment
    experiment_path = mkexperiment(config, cover=True)
    save_inter_result = os.path.join(experiment_path, 'inter_result')
    model_path = os.path.join(config.model_path, config.name)

    # load data
    base_path = get_base_path()
    datasets, meta = load_training_volume({"source": config.source_path, "mask": config.mask_path, "label": config.lable_path})

    # model: BF
    bf_network = UNet(1, config.n_layers, config.starting_filters, 3, config.kernel_initializer, config.batch_norm,
                      0., get_act_function(config.act_func), config.conv_per_layer, False, False, None)

    # bf_network.load_weights(ckp_path + "zdir_calc-HRbf-rmse-weights")
    bf_network.summary((256, 256, 256, 1))  # (64, 64, 64, 1)


    net_analysis(bf_network)

    # cost function
    loss_fn = tf.keras.losses.MeanSquaredError()

    # optimizer
    optimizer = tf.keras.optimizer.Adam(learning_rate=config.learning_rate)

    # setup device
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:' + config.GPU_NUM:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    # train





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment info
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--experiment_path', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--source_path', type=str, default='')
    parser.add_argument('--mask_path', type=str, default='')
    parser.add_argument('--GPU_NUM', type=str, default='6')

    # model hyper-parameters
    parser.add_argument('--OUTPUT_C', type=int, default=1)  # OUTPUT CHANNELS
    parser.add_argument('--n_layers', type=int, default=7)
    parser.add_argument('--starting_filters', type=int, default=16)
    parser.add_argument('--kernel_initializer', type=str, default='he_normal')
    parser.add_argument('--batch_norm', )

    parser.add_argument('--act_func', type=str, default='relu')
    parser.add_argument('--conv_per_layer', type=int, default=1)

    # training hyper-parameters
    parser.add_argument('--all_batch_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0003)

    parser.add_argument('--weight_vn', type=float, default=10.0)
    parser.add_argument('--vn_n_steps', type=int, default=6)
    parser.add_argument('--vn_batch_norm', type=bool, default=False)
    parser.add_argument('--vn_batch_size', type=int, default=1)
    parser.add_argument('--vn_lr', type=float, default=0.005)
    parser.add_argument('--vn_n_layers', type=int, default=6)
    parser.add_argument('--vn_kernel_initializer', type=str, default='he_normal')
    parser.add_argument('--vn_act_func', type=str, default='relu')
    parser.add_argument('--vn_l_init', type=float, default=0.1)
    parser.add_argument('--vn_starting_filters', type=int, default=16)
    parser.add_argument('--vn_dt_loss', type=str, default='RMSE')

    # misc
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--result_path', type=str, default='./results/')

    config = parser.parse_args()

    config.source_path = config.data_dir + 'source'
    config.source_path = config.data_dir + 'mask'

    main(config)
