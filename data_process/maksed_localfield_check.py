# -*- coding: utf-8 -*-
'''

 @Time   : 4/19/23, 3:06 PM
 @Author : Jie
 
'''
import os

path_localfield2 = '/DATA_Inter/cj/localfield_100_400_50_8/'
path_localfield = '/DATA_Temp/cj/QSM/NeXtQSM/train_localfield/'
path_mask = '/DATA_Temp/cj/QSM/NeXtQSM/mask/'

path_dir = '/DATA_Temp/cj/QSM/NeXtQSM/train_localfield_masked/'

num_samples = 2000

file_list = os.listdir(path_dir)
num = 0

for n in range(num_samples):
    file_name = path_dir + 'localfield_' + str(n) + '.nii.gz'

    if os.path.exists(file_name):
        continue
    else:
        num = num + 1
        print(n)

print(num)
