# -*- coding: utf-8 -*-
'''

 @Time   : 4/20/23, 10:52 PM
 @Author : Jie
 
'''

import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info

print(tf.__version__)

# Get the build information about Tensorflow
print(tf_build_info.cuda_version_number)
print(tf_build_info.cudnn_version_number)
print(tf.test.is_built_with_cuda())
# print(tf.test.is_built_with_cudnn())

print("Tensorflow is using GPU: ", tf.test.is_gpu_available())

print(tf.config.experimental.list_physical_devices('GPU'))
tf.debugging.set_log_device_placement(True)

print("Tensorflow is using GPU: ", tf.test.is_gpu_available())
