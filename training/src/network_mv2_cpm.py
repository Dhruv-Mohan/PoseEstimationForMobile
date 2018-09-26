# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import functools
from network_base import max_pool, upsample, inverted_bottleneck, separable_conv, convb, is_trainable
import tensorflow.contrib.layers as layers

from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet as lib
from nets.mobilenet import mobilenet_v2

slim = tf.contrib.slim
op = lib.op

N_KPOINTS = 91
STAGE_NUM = 3
_SEP_CHANNELS_ = 256 #512
_CPM_CHANNELS_ = 96 #128

'''
expanded_conv(input_tensor,
                  num_outputs,
                  expansion_size=expand_input_by_factor(6),
                  stride=1,
                  rate=1,
                  kernel_size=(3, 3),
                  residual=True,
                  normalizer_fn=None,
                  split_projection=1,
                  split_expansion=1,
                  expansion_transform=None,
                  depthwise_location='expansion',
                  depthwise_channel_multiplier=1,
                  endpoints=None,
                  use_explicit_padding=False,
                  padding='SAME',
                  scope=None)

'''

def build_cpm(input_):
                #STAGE 1
                conv4_3_CPM = ops.expanded_conv(input_, _CPM_CHANNELS_, expansion_size=2)
                conv4_4_CPM = ops.expanded_conv(conv4_3_CPM, _CPM_CHANNELS_, expansion_size=2)
                #conv4_5_CPM = ops.expanded_conv(conv4_4_CPM, _CPM_CHANNELS_, expansion_size=2)
                conv4_6_CPM = ops.expanded_conv(conv4_4_CPM, _CPM_CHANNELS_, expansion_size=2)
                conv4_7_CPM = ops.expanded_conv(conv4_6_CPM, _CPM_CHANNELS_, expansion_size=2)

                conv5_1_CPM = layers.separable_conv2d(conv4_7_CPM, _SEP_CHANNELS_, kernel_size=1, scope='conv5_1_CPM', depth_multiplier=1)
                #conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
                conv5_2_CPM = layers.separable_conv2d(conv5_1_CPM, 91, kernel_size=1, scope='conv5_2_CPM', depth_multiplier=1, activation_fn=None)

                concat_stage2 = tf.concat(axis=3, values=[conv5_2_CPM, conv4_7_CPM])
                #STAGE 2
                Mconv1_stage2 = ops.expanded_conv(concat_stage2, _CPM_CHANNELS_, expansion_size=1, kernel_size=(7, 7))
                Mconv2_stage2 = ops.expanded_conv(Mconv1_stage2, _CPM_CHANNELS_, expansion_size=2, kernel_size=(7, 7))
                #Mconv3_stage2 = ops.expanded_conv(Mconv2_stage2, _CPM_CHANNELS_, expansion_size=2, kernel_size=(7, 7))
                Mconv4_stage2 = ops.expanded_conv(Mconv2_stage2, _CPM_CHANNELS_, expansion_size=2, kernel_size=(3, 3))
                Mconv5_stage2 = ops.expanded_conv(Mconv4_stage2, _CPM_CHANNELS_, expansion_size=2, kernel_size=(3, 3))

                Mconv6_stage2 = layers.separable_conv2d(Mconv5_stage2, 128, kernel_size=1, scope='Mconv6_stage2', depth_multiplier=1)
                #Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
                Mconv7_stage2 = layers.separable_conv2d(Mconv6_stage2, 91, kernel_size=1, scope='Mconv7_stage2', depth_multiplier=1, activation_fn=None)

                concat_stage3 = tf.concat(axis=3, values=[Mconv7_stage2, conv4_7_CPM])
                #STAGE 3
                Mconv1_stage3 = ops.expanded_conv(concat_stage3, _CPM_CHANNELS_, expansion_size=1, kernel_size=(7, 7))
                Mconv2_stage3 = ops.expanded_conv(Mconv1_stage3, _CPM_CHANNELS_, expansion_size=2, kernel_size=(7, 7))
                #Mconv3_stage3 = ops.expanded_conv(Mconv2_stage3, _CPM_CHANNELS_, expansion_size=2, kernel_size=(7, 7))
                Mconv4_stage3 = ops.expanded_conv(Mconv2_stage3, _CPM_CHANNELS_, expansion_size=2, kernel_size=(3, 3))
                Mconv5_stage3 = ops.expanded_conv(Mconv4_stage3, _CPM_CHANNELS_, expansion_size=2, kernel_size=(3, 3))

                Mconv6_stage3 = layers.separable_conv2d(Mconv5_stage3, 128, kernel_size=1, scope='Mconv6_stage3', depth_multiplier=1)
                #Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
                Mconv7_stage3 = layers.separable_conv2d(Mconv6_stage3, 91, kernel_size=1, scope='Mconv7_stage3', depth_multiplier=1, activation_fn=None)

                return conv5_2_CPM, Mconv7_stage2, Mconv7_stage3



def build_network(input_, trainable):
        is_trainable(trainable)
        if trainable:
                with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
                    logits, endpoints = mobilenet_v2.mobilenet(input_, num_classes=91, depth_multiplier=1.4, final_endpoint='layer_7')
                conv5_2_CPM, Mconv7_stage2, Mconv7_stage3 = build_cpm(endpoints['layer_7/output'])

        else:

                logits, endpoints = mobilenet_v2.mobilenet(input_, num_classes=91, depth_multiplier=1.4, final_endpoint='layer_7')
                conv5_2_CPM, Mconv7_stage2, Mconv7_stage3 = build_cpm(endpoints['layer_7/output'])

        for vari in endpoints:
                print(vari + ' Shape={}'.format(endpoints[vari].get_shape().as_list()))



        #input(endpoints)
        #for variable in endpoints:
        #conv = endpoints['layer_5/expansion_output'] #64x64x192
        #conv = endpoints['layer_7/depthwise_output']  # 32x32x228

        #pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
        '''
        conv4_1 = layers.conv2d(conv, 512, 3, 1, activation_fn=None, scope='conv4_1')
        conv4_1 = tf.nn.relu(conv4_1)
        conv4_2 = layers.conv2d(conv4_1, 512, 3, 1, activation_fn=None, scope='conv4_2')
        conv4_2 = tf.nn.relu(conv4_2)
        #CPM STAGES
        conv4_3_CPM = layers.conv2d(conv, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')
        conv4_3_CPM = tf.nn.relu(conv4_3_CPM)
        conv4_4_CPM = layers.conv2d(conv4_3_CPM, 256, 3, 1, activation_fn=None, scope='conv4_4_CPM')
        conv4_4_CPM = tf.nn.relu(conv4_4_CPM)
        conv4_5_CPM = layers.conv2d(conv4_4_CPM, 256, 3, 1, activation_fn=None, scope='conv4_5_CPM')
        conv4_5_CPM = tf.nn.relu(conv4_5_CPM)
        conv4_6_CPM = layers.conv2d(conv4_5_CPM, 256, 3, 1, activation_fn=None, scope='conv4_6_CPM')
        conv4_6_CPM = tf.nn.relu(conv4_6_CPM)
        conv4_7_CPM = layers.conv2d(conv4_6_CPM, 128, 3, 1, activation_fn=None, scope='conv4_7_CPM')
        conv4_7_CPM = tf.nn.relu(conv4_7_CPM)
        conv5_1_CPM = layers.conv2d(conv4_7_CPM, 512, 1, 1, activation_fn=None, scope='conv5_1_CPM')
        conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
        conv5_2_CPM = layers.conv2d(conv5_1_CPM, 91, 1, 1, activation_fn=None, scope='conv5_2_CPM')
        concat_stage2 = tf.concat(axis=3, values=[conv5_2_CPM, conv4_7_CPM])
        Mconv1_stage2 = layers.conv2d(concat_stage2, 128, 7, 1, activation_fn=None, scope='Mconv1_stage2')
        Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
        Mconv2_stage2 = layers.conv2d(Mconv1_stage2, 128, 7, 1, activation_fn=None, scope='Mconv2_stage2')
        Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
        Mconv3_stage2 = layers.conv2d(Mconv2_stage2, 128, 7, 1, activation_fn=None, scope='Mconv3_stage2')
        Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
        Mconv4_stage2 = layers.conv2d(Mconv3_stage2, 128, 7, 1, activation_fn=None, scope='Mconv4_stage2')
        Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
        Mconv5_stage2 = layers.conv2d(Mconv4_stage2, 128, 7, 1, activation_fn=None, scope='Mconv5_stage2')
        Mconv5_stage2 = tf.nn.relu(Mconv5_stage2)
        Mconv6_stage2 = layers.conv2d(Mconv5_stage2, 128, 1, 1, activation_fn=None, scope='Mconv6_stage2')
        Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
        Mconv7_stage2 = layers.conv2d(Mconv6_stage2, 91, 1, 1, activation_fn=None, scope='Mconv7_stage2')
        concat_stage3 = tf.concat(axis=3, values=[Mconv7_stage2, conv4_7_CPM])
        Mconv1_stage3 = layers.conv2d(concat_stage3, 128, 7, 1, activation_fn=None, scope='Mconv1_stage3')
        Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
        Mconv2_stage3 = layers.conv2d(Mconv1_stage3, 128, 7, 1, activation_fn=None, scope='Mconv2_stage3')
        Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
        Mconv3_stage3 = layers.conv2d(Mconv2_stage3, 128, 7, 1, activation_fn=None, scope='Mconv3_stage3')
        Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
        Mconv4_stage3 = layers.conv2d(Mconv3_stage3, 128, 7, 1, activation_fn=None, scope='Mconv4_stage3')
        Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
        Mconv5_stage3 = layers.conv2d(Mconv4_stage3, 128, 7, 1, activation_fn=None, scope='Mconv5_stage3')
        Mconv5_stage3 = tf.nn.relu(Mconv5_stage3)
        Mconv6_stage3 = layers.conv2d(Mconv5_stage3, 128, 1, 1, activation_fn=None, scope='Mconv6_stage3')
        Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
        Mconv7_stage3 = layers.conv2d(Mconv6_stage3, 91, 1, 1, activation_fn=None, scope='Mconv7_stage3')
        
        '''

        return conv5_2_CPM, Mconv7_stage2, Mconv7_stage3
