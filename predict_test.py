# -*- coding: UTF-8 -*-
'''
Created on  May 05 10:39 AM 2023

@author: cj
'''

import argparse
import os
import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy.io import savemat

from ext.lab2im import utils


def test(config):
    # check gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # random seed
    np.random.seed(2)

    # model path
    model_path_bf = config.model_path_bf + '_wholemodel/' + config.name_bf + '/' + config.name_bf + '_epoch_' + config.model_num_bf
    model_path_vn = config.model_path_vn + '_wholemodel/' + config.name_vn + '/' + config.name_vn + '_epoch_' + config.model_num_vn

    if not os.path.exists(model_path_bf):
        print('Model not found, please check you path to model')
        print(model_path_bf)
        os._exit(0)
    if not os.path.exists(model_path_vn):
        print('Model not found, please check you path to model')
        print(model_path_vn)
        os._exit(0)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # load model
    bf_network = tf.keras.models.load_model(model_path_bf)
    vn_network = tf.keras.models.load_model(model_path_vn)

    # loop
    predict_bf_list = []
    bf_gt_list = []
    predict_vn_list = []
    vn_gt_list = []

    # load data
    for filename in os.listdir(config.bf_input_path):
        name = filename.split('_')[1]
        bf_input_nii = nib.load(config.bf_input_path + '/totalfield_' + name)
        aff = bf_input_nii.affine
        header = bf_input_nii.header
        bf_input = bf_input_nii.get_fdata()
        # bf_gt = nib.load(config.bf_gt_path + '/localfield_' + name).get_fdata()  # localfield
        # vn_gt = nib.load(config.vn_gt_path + '/image_' + name).get_fdata()  # chimap

        # bf_gt_list.append(bf_gt)
        # vn_gt_list.append(vn_gt)

        # expand dims
        bf_input = np.expand_dims(bf_input, -1)
        bf_input = np.expand_dims(bf_input, 0)

        predict_bf = bf_network.predict(bf_input)
        predict_vn = vn_network.predict(predict_bf)

        # squeeze dims
        predict_bf = np.squeeze(predict_bf)
        predict_vn = np.squeeze(predict_vn)

        # predict_bf_list.append(predict_bf)
        predict_vn_list.append(predict_vn)

        # save results: singular
        utils.save_volume(predict_bf, aff, header,
                          os.path.join(config.result_path, config.test_dir,
                                       config.test_dir + '_localfield_' + config.name_bf + '_' + name))
        utils.save_volume(predict_vn, aff, header,
                          os.path.join(config.result_path, config.test_dir,
                                       config.test_dir + '_chimap_' + config.name_vn + '_' + name))
        print('Save: ', name.split('.')[0])

    # save results: fixed, with label
    # predict_bf_array = np.array(predict_bf_list)
    predict_vn_array = np.array(predict_vn_list)
    # bf_gt_array = np.array(bf_gt_list)
    # vn_gt_array = np.array(vn_gt_list)
    print('.'*30)
    print('Predict result: ', predict_vn_array.shape)
    # savemat(os.path.join(config.result_path, config.test_dir + '_localfield_' + config.name_bf + '.mat'),
    #         {
    #             'output': predict_bf_array,
    #             'label': bf_gt_array
    #         })
    # savemat(os.path.join(config.result_path, config.test_dir + '_chimap_' + config.name_vn + '.mat'),
    #         {
    #             'output': predict_vn_array,
    #             'label': vn_gt_array
    #         })
    # print('Save result in ', config.test_dir + '_localfield_' + config.name_bf + '.mat')
    # print('Save result in ', config.test_dir + '_chimap_' + config.name_vn + '.mat')
    # print('.'*30)
    print('Finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment info
    parser.add_argument('--bf_input_path', type=str, default='./data/test_totalfield/')
    parser.add_argument('--bf_gt_path', type=str, default='./data/test_localfield/')
    parser.add_argument('--vn_gt_path', type=str, default='./data/test_chimap/')
    parser.add_argument('--GPU_NUM', type=str, default='7')  # 3[0], 4[2], 5[4], 6[5], 7[6]

    # misc
    parser.add_argument('--name_bf', type=str, default='v2_7layer')
    parser.add_argument('--name_vn', type=str, default='v2_7layer')
    parser.add_argument('--model_path_bf', type=str, default='./models/bf')  # total_field to local_field
    parser.add_argument('--model_path_vn', type=str, default='./models/vn')
    parser.add_argument('--model_num_bf', type=str, default='12')
    parser.add_argument('--model_num_vn', type=str, default='2')

    parser.add_argument('--result_path', type=str, default='./predict_result/')
    parser.add_argument('--test_dir', type=str, default='version1_simu_best')

    config = parser.parse_args()
    test(config)
