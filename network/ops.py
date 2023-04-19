# -*- coding: utf-8 -*-
'''

 @Time   : 4/19/23, 4:41 PM
 @Author : Jie
 
'''

from tensorflow.keras import layers

def net_analysis(net):
    tol_conv = 0
    print('.'*70)
    for layer in net.named_modules():
        if isinstance(layer[1], layers.Conv3d):
            print(layer[1])
            tol_conv += 1
    print('.' * 70)
    print('# Model contains %d Conv layers.'%(tol_conv))
    print('# Model parameters:', sum(param.numel() for param in net.parameters()))
    print('.' * 70)
