# -*- coding: utf-8 -*-
# @Time    : 18-3-6 3:20 PM
# @Author  : zengzihua@huya.com
# @FileName: data_filter.py
# @Software: PyCharm

import network_mv2_cpm
import network_mv2_hourglass

def get_network(type, input, trainable=True):
    if type == 'mv2_cpm':
        conv5_2_CPM, Mconv7_stage2, Mconv7_stage3, conv5_2_CPMl, Mconv7_stage2l, Mconv7_stage3l, out = network_mv2_cpm.build_network(input,  trainable)
        #return net, loss
    elif type == "mv2_hourglass":
        net, loss = network_mv2_hourglass.build_network(input, trainable)        
    return conv5_2_CPM, Mconv7_stage2, Mconv7_stage3, conv5_2_CPMl, Mconv7_stage2l, Mconv7_stage3l, out
