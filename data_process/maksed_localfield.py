# -*- coding: utf-8 -*-
'''

 @Time   : 4/19/23, 3:06 PM
 @Author : Jie
 
'''
import os
import time
import numpy as np
import nibabel as nib

from ext.lab2im import utils

path_localfield2 = '/DATA_Inter/cj/localfield_100_400_50_8/'
path_localfield = '/DATA_Temp/cj/QSM/NeXtQSM/train_localfield/'
path_mask = '/DATA_Temp/cj/QSM/NeXtQSM/mask/'

path_dir = '/DATA_Temp/cj/QSM/NeXtQSM/train_localfield_masked/'

num_samples = 2000

for n in range(num_samples):
    # start = time.clock()
    local_field_nii = nib.load(path_localfield2 + 'localfield_' + str(n) + '.nii.gz')
    aff = local_field_nii.affine
    header = local_field_nii.header

    local_field_nomask = local_field_nii.get_data()
    mask = nib.load(path_mask + 'mask_' + str(n) + '.nii.gz').get_data()

    local_field_masked = np.multiply(local_field_nomask, mask)

    # num of zero
    # print('localfiled: ',np.sum(local_field_nomask == 0))
    # print('mask: ', np.sum(mask == 0))
    # print('localfiled_masked: ', np.sum(local_field_masked == 0))

    utils.save_volume(local_field_masked, aff, header,
                      os.path.join(path_dir, 'localfield_%s.nii.gz' % n))

    # end = time.clock()

    # print('Compeleted: ', str(n), '  Time: ',str(end - start))
    print(n)