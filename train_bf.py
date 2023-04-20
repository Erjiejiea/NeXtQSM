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
from utils.data_loader import load_training_volume, load_testing_volume
from utils.misc import mkexperiment, get_act_function
from utils.save_image import SaveImageCallback


def main(config):
    # check gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM

    # random seed
    np.random.seed(2)

    # set up experiment
    experiment_path = mkexperiment(config, cover=True)
    inter_result_path = os.path.join(experiment_path, 'inter_result')
    model_path = os.path.join(config.model_path, config.name)
    bf_checkpoint_path = model_path + 'bf_{epoch_04d}.ckpt'

    # load data
    totalfield_paths = os.listdir(config.totalfield_path)
    localfield_paths = os.listdir(config.localfield_path)
    sim_datasets = []
    sim_metas = []
    num_samples = 0
    # for index in range(len(totalfield_paths)):
    for index in range(2):
        sim_dataset, sim_meta = load_training_volume(
        {"totalfiled": os.path.join(config.totalfield_path, totalfield_paths[index]),
         "localfiled": os.path.join(config.localfield_path, localfield_paths[index]),
         "mask": None})
        sim_datasets.append(sim_dataset)
        sim_metas.append(sim_meta)
        num_samples = num_samples + 1
    print('Loaded {} samples for training.'.format(num_samples))

    real_tot_paths = os.listdir(os.path.join(config.realdata_path, 'totalfield'))
    real_loc_paths = os.listdir(os.path.join(config.realdata_path, 'localfield'))
    real_datasets = []
    real_metas = []
    num_samples_real = 0
    for index2 in range(2):
        real_dataset, real_meta = load_testing_volume(
            {"totalfiled": os.path.join(config.realdata_path, 'totalfield', real_tot_paths[index2]),
             "localfiled": os.path.join(config.realdata_path, 'localfield', real_loc_paths[index2]),
             "mask": None})
        real_datasets.append(real_dataset)
        real_metas.append(real_meta)
        num_samples_real = num_samples_real + 1
    print('Loded {} real brain.'.format(num_samples_real))

    # model: BF
    bf_network = UNet(1, config.n_layers, config.starting_filters, 3, config.kernel_initializer, config.batch_norm,
                      0., get_act_function(config.act_func), config.conv_per_layer, False, False, None)
    # bf_network.load_weights(ckp_path + "zdir_calc-HRbf-rmse-weights")
    bf_network.summary((256, 256, 256, 1))  # (64, 64, 64, 1)

    # cost function
    loss_fn = tf.keras.losses.MeanSquaredError()

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.09, beta_2=0.009)

    # setup device
    # device_name = tf.test.gpu_device_name()
    # print(device_name)
    # if device_name != '/device:GPU:' + config.GPU_NUM:
    #     raise SystemError('GPU device not found')
    # print('Found GPU at: {}'.format(device_name))

    # train
    bf_network.compile(loss=loss_fn, optimizer=optimizer)

    # images = tf.expand_dims(sim_bf_datasets, 4)
    # labels = tf.expand_dims(sim_sus_datasets, 4)
    images = tf.convert_to_tensor(np.stack(item["totalfield"] for item in sim_datasets))
    labels = tf.convert_to_tensor(np.stack(item["localfield"] for item in sim_datasets))

    train_sample_num = 2000 * 0.8
    test_sample_num= 2000 * 0.1
    val_sample_num = 2000 * 0.1
    train_images, val_images, test_images = tf.split(
        images,
        num_or_size_splits=[train_sample_num, val_sample_num, test_sample_num],
        # axis=0
    )
    train_labels, val_labels, test_lables = tf.split(
        labels,
        num_or_size_splits=[train_sample_num, val_sample_num, test_sample_num],
        # axis=0
    )

    # create checkpoint callback
    real_data = tf.convert_to_tensor(np.stack(item["totalfield"] for item in real_datasets))
    cp_callpack = SaveImageCallback(save_dir_model=bf_checkpoint_path,
                                    save_dir_inter_result=inter_result_path,
                                    interval=config.save_epoch,
                                    real_data=real_data)

    print('# Fit bf_model on training data')
    bf_history = bf_network.fit(train_images,
                                train_labels,
                                batch_size=config.batch_size,
                                epochs=config.epochs_train,
                                verbose=2,
                                callbacks=[cp_callpack],
                                shuffle=True,
                                validation_data=(val_images, val_labels))  # pass callback to training for saving the model

    loss_bf_history = bf_history.history['loss']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment info
    parser.add_argument('--name', type=str, default='version1')
    parser.add_argument('--experiment_path', type=str, default='')
    parser.add_argument('--sus_path', type=str, default='/DATA_Temp/cj/QSM/NeXtQSM/train/train_synthetic_brain/')
    parser.add_argument('--localfield_path', type=str, default='/DATA_Temp/cj/QSM/NeXtQSM/train_localfield_masked/')
    parser.add_argument('--totalfield_path', type=str, default='/DATA_Temp/cj/QSM/NeXtQSM/train_totalfield/')
    parser.add_argument('--realdata_path', type=str, default='/DATA_Temp/cj/QSM/NeXtQSM/realdata_for_NeXtQSM/')
    parser.add_argument('--GPU_NUM', type=str, default='3')

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
    parser.add_argument('--learning_rate', type=float, default=4e-4)  # .0003

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

    parser.add_argument('--epochs_train', type=int, default=100)
    parser.add_argument('--save_period', type=int, default=1)

    # misc
    parser.add_argument('--model_path', type=str, default='./models/')  # phase unwrapped (totalfield) to phase tissue (localfield)
    # parser.add_argument('--vn_model_path', type=str, default='./models/vn/')  # phase tissue (localfield) to susceptibility (chimap)
    parser.add_argument('--result_path', type=str, default='./results/')

    config = parser.parse_args()
    main(config)
