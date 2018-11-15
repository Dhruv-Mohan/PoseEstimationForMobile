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

from network_base import max_pool, upsample, inverted_bottleneck, separable_conv, convb, is_trainable
import tensorflow.contrib.layers as layers
N_KPOINTS = 91
STAGE_NUM = 3
_SEP_CHANNELS_ = 256  # 512
_CPM_CHANNELS_ = 96  # 128

from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet as lib
from nets.mobilenet import mobilenet_v2
out_channel_ratio = lambda d: max(int(d * 1.4), 8)
up_channel_ratio = lambda d: int(d * 1.)
out_channel_cpm = lambda d: max(int(d * 0.75), 8)


def build_cpm(input_):

    # STAGE 1
    stage_0_bottleneck = slim.stack(input_, inverted_bottleneck,
                              [
                                  (up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                                  (up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                                  (up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                                  (up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                              ], scope="cpm_stage_0_bottleneck")
    '''
    stage_0_sepconv = slim.stack(stage_0_bottleneck, separable_conv,
                   [
                       (_SEP_CHANNELS_, 1, 1),
                       (N_KPOINTS, 1, 1)
                   ], scope='cpm_stage_0_sep_conv')
    '''
    conv5_1_CPM = layers.separable_conv2d(stage_0_bottleneck, _SEP_CHANNELS_, kernel_size=1, scope='conv5_1_CPM',
                                          depth_multiplier=1,
                                          weights_initializer=tf.contrib.layers.xavier_initializer())
    # conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
    stage_0_sepconv = layers.separable_conv2d(conv5_1_CPM, N_KPOINTS, kernel_size=1, scope='conv5_2_CPM', depth_multiplier=1,
                                          activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer())

    concat_stage2 = tf.concat(axis=3, values=[stage_0_bottleneck, stage_0_sepconv])

    stage_1_bottleneck = slim.stack(concat_stage2, inverted_bottleneck,
                              [
                                  (up_channel_ratio(1), _CPM_CHANNELS_, 0, 7),
                                  (up_channel_ratio(2), _CPM_CHANNELS_, 0, 7),
                                  (up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                                  (up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                                  #(up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                              ], scope="cpm_stage_1_bottleneck")
    '''
    stage_1_sepconv = slim.stack(stage_1_bottleneck, separable_conv,
                   [
                       (128, 1, 1),
                       (N_KPOINTS, 1, 1)
                   ], scope='cpm_stage_1_sep_conv')
    '''

    Mconv6_stage2 = layers.separable_conv2d(stage_1_bottleneck, 128, kernel_size=1, scope='Mconv6_stage2',
                                            depth_multiplier=1)
    # Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
    stage_1_sepconv = layers.separable_conv2d(Mconv6_stage2, N_KPOINTS, kernel_size=1, scope='Mconv7_stage2', depth_multiplier=1,
                                            activation_fn=None)

    concat_stage3 = tf.concat(axis=3, values=[stage_1_bottleneck, stage_1_sepconv])

    stage_2_bottleneck = slim.stack(concat_stage3, inverted_bottleneck,
                                    [
                                        (up_channel_ratio(1), _CPM_CHANNELS_, 0, 7),
                                        (up_channel_ratio(2), _CPM_CHANNELS_, 0, 7),
                                        (up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                                        (up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                                        #(up_channel_ratio(2), _CPM_CHANNELS_, 0, 3),
                                    ], scope="cpm_stage_2_bottleneck")
    '''
    stage_2_sepconv = slim.stack(stage_2_bottleneck, separable_conv,
                                 [
                                     (128, 1, 1),
                                     (N_KPOINTS, 1, 1)
                                 ], scope='cpm_stage_2_sep_conv')
    '''
    Mconv6_stage3 = layers.separable_conv2d(stage_2_bottleneck, 128, kernel_size=1, scope='Mconv6_stage3',
                                            depth_multiplier=1)
    # Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
    stage_2_sepconv = layers.separable_conv2d(Mconv6_stage3, N_KPOINTS, kernel_size=1, scope='Mconv7_stage3', depth_multiplier=1,
                                            activation_fn=None)

    return stage_0_sepconv, stage_1_sepconv, stage_2_sepconv
    '''
    # STAGE 1
    conv4_3_CPM = ops.expanded_conv(input_, _CPM_CHANNELS_, expansion_size=2)
    conv4_4_CPM = ops.expanded_conv(conv4_3_CPM, _CPM_CHANNELS_, expansion_size=2)
    # conv4_5_CPM = ops.expanded_conv(conv4_4_CPM, _CPM_CHANNELS_, expansion_size=2)
    conv4_6_CPM = ops.expanded_conv(conv4_4_CPM, _CPM_CHANNELS_, expansion_size=2)
    conv4_7_CPM = ops.expanded_conv(conv4_6_CPM, _CPM_CHANNELS_, expansion_size=2)
    conv5_1_CPM = layers.separable_conv2d(conv4_7_CPM, _SEP_CHANNELS_, kernel_size=1, scope='conv5_1_CPM',
                                          depth_multiplier=1,
                                          weights_initializer=tf.contrib.layers.xavier_initializer())
    # conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
    conv5_2_CPM = layers.separable_conv2d(conv5_1_CPM, 91, kernel_size=1, scope='conv5_2_CPM', depth_multiplier=1,
                                          activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer())
    concat_stage2 = tf.concat(axis=3, values=[conv5_2_CPM, conv4_7_CPM])
    # STAGE 2
    Mconv1_stage2 = ops.expanded_conv(concat_stage2, _CPM_CHANNELS_, expansion_size=1, kernel_size=(7, 7))
    Mconv2_stage2 = ops.expanded_conv(Mconv1_stage2, _CPM_CHANNELS_, expansion_size=2, kernel_size=(7, 7))
    # Mconv3_stage2 = ops.expanded_conv(Mconv2_stage2, _CPM_CHANNELS_, expansion_size=2, kernel_size=(7, 7))
    Mconv4_stage2 = ops.expanded_conv(Mconv2_stage2, _CPM_CHANNELS_, expansion_size=2, kernel_size=(3, 3))
    Mconv5_stage2 = ops.expanded_conv(Mconv4_stage2, _CPM_CHANNELS_, expansion_size=2, kernel_size=(3, 3))
    Mconv6_stage2 = layers.separable_conv2d(Mconv5_stage2, 128, kernel_size=1, scope='Mconv6_stage2',
                                            depth_multiplier=1)
    # Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
    Mconv7_stage2 = layers.separable_conv2d(Mconv6_stage2, 91, kernel_size=1, scope='Mconv7_stage2', depth_multiplier=1,
                                            activation_fn=None)
    concat_stage3 = tf.concat(axis=3, values=[Mconv7_stage2, conv4_7_CPM])
    # STAGE 3
    Mconv1_stage3 = ops.expanded_conv(concat_stage3, _CPM_CHANNELS_, expansion_size=1, kernel_size=(7, 7))
    Mconv2_stage3 = ops.expanded_conv(Mconv1_stage3, _CPM_CHANNELS_, expansion_size=2, kernel_size=(7, 7))
    # Mconv3_stage3 = ops.expanded_conv(Mconv2_stage3, _CPM_CHANNELS_, expansion_size=2, kernel_size=(7, 7))
    Mconv4_stage3 = ops.expanded_conv(Mconv2_stage3, _CPM_CHANNELS_, expansion_size=2, kernel_size=(3, 3))
    Mconv5_stage3 = ops.expanded_conv(Mconv4_stage3, _CPM_CHANNELS_, expansion_size=2, kernel_size=(3, 3))
    Mconv6_stage3 = layers.separable_conv2d(Mconv5_stage3, 128, kernel_size=1, scope='Mconv6_stage3',
                                            depth_multiplier=1)
    # Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
    Mconv7_stage3 = layers.separable_conv2d(Mconv6_stage3, 91, kernel_size=1, scope='Mconv7_stage3', depth_multiplier=1,
                                            activation_fn=None)
    return conv5_2_CPM, Mconv7_stage2, Mconv7_stage3
    '''
    return conv5_2_CPM, Mconv7_stage2, Mconv7_stage3


def build_network(input, trainable):
    is_trainable(trainable)

    net = convb(input, 3, 3, out_channel_ratio(32), 2, name="Conv2d_0")

    with tf.variable_scope('MobilenetV2'):

        # 128, 112
        mv2_branch_0 = slim.stack(net, inverted_bottleneck,
                                  [
                                      (1, out_channel_ratio(16), 0, 3),
                                      (1, out_channel_ratio(16), 0, 3)
                                  ], scope="MobilenetV2_part_0")

        mv2_branch_1 = slim.stack(mv2_branch_0, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(24), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(24), 0, 3),

                                  ], scope="MobilenetV2_part_1")

        # 32, 28
        mv2_branch_2 = slim.stack(mv2_branch_1, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(32), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                  ], scope="MobilenetV2_part_2")
        '''
        # 64, 56
        mv2_branch_1 = slim.stack(mv2_branch_0, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(24), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                  ], scope="MobilenetV2_part_1")

        # 32, 28
        mv2_branch_2 = slim.stack(mv2_branch_1, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(32), 1, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                  ], scope="MobilenetV2_part_2")
                                  


        # 16, 14
        mv2_branch_3 = slim.stack(mv2_branch_2, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(64), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(64), 0, 3)
                                  ], scope="MobilenetV2_part_3")

        # 8, 7
        mv2_branch_4 = slim.stack(mv2_branch_3, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(96), 1, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                      #(up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                  ], scope="MobilenetV2_part_4")

        '''

    with tf.variable_scope("Convolutional_Pose_Machine"):
        conv5_2_CPM, Mconv7_stage2, Mconv7_stage3 = build_cpm(mv2_branch_2)
        return conv5_2_CPM, Mconv7_stage2, Mconv7_stage3

        l2s = []
        prev = None
        for stage_number in range(STAGE_NUM):
            if prev is not None:
                inputs = tf.concat([cancat_mv2, prev], axis=3)
            else:
                inputs = cancat_mv2

            kernel_size = 7
            lastest_channel_size = 128
            if stage_number == 0:
                kernel_size = 3
                lastest_channel_size = 512

            _ = slim.stack(inputs, inverted_bottleneck,
                           [
                               (2, out_channel_cpm(32), 0, kernel_size),
                               (up_channel_ratio(4), out_channel_cpm(32), 0, kernel_size),
                               (up_channel_ratio(4), out_channel_cpm(32), 0, kernel_size),
                           ], scope="stage_%d_mv2" % stage_number)

            _ = slim.stack(_, separable_conv,
                           [
                               (out_channel_ratio(lastest_channel_size), 1, 1),
                               (N_KPOINTS, 1, 1)
                           ], scope="stage_%d_mv1" % stage_number)

            prev = _
            cpm_out = upsample(_, 1, "stage_%d_out" % stage_number)
            l2s.append(cpm_out)

    return cpm_out, l2s


def build_network2(input, trainable):
    is_trainable(trainable)

    net = convb(input, 3, 3, out_channel_ratio(32), 2, name="Conv2d_0")

    with tf.variable_scope('MobilenetV2'):

        # 128, 112
        mv2_branch_0 = slim.stack(net, inverted_bottleneck,
                                  [
                                      (1, out_channel_ratio(16), 0, 3),
                                      (1, out_channel_ratio(16), 0, 3)
                                  ], scope="MobilenetV2_part_0")
        '''
        mv2_branch_1 = slim.stack(mv2_branch_0, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(24), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),

                                  ], scope="MobilenetV2_part_1")

        # 32, 28
        mv2_branch_2 = slim.stack(mv2_branch_1, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(32), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                  ], scope="MobilenetV2_part_2")
        '''
        # 64, 56
        mv2_branch_1 = slim.stack(mv2_branch_0, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(24), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                  ], scope="MobilenetV2_part_1")

        # 32, 28
        mv2_branch_2 = slim.stack(mv2_branch_1, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(32), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                  ], scope="MobilenetV2_part_2")
        
        # 16, 14
        mv2_branch_3 = slim.stack(mv2_branch_2, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(64), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                  ], scope="MobilenetV2_part_3")

        # 8, 7
        mv2_branch_4 = slim.stack(mv2_branch_3, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(96), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(96), 0, 3)
                                  ], scope="MobilenetV2_part_4")

        cancat_mv2 = tf.concat(
            [
                max_pool(mv2_branch_0, 4, 4, 4, 4, name="mv2_0_max_pool"),
                max_pool(mv2_branch_1, 2, 2, 2, 2, name="mv2_1_max_pool"),
                mv2_branch_2,
                upsample(mv2_branch_3, 2, name="mv2_3_upsample"),
                upsample(mv2_branch_4, 4, name="mv2_4_upsample")
            ]
            , axis=3)

    with tf.variable_scope("Convolutional_Pose_Machine"):
        conv5_2_CPM, Mconv7_stage2, Mconv7_stage3 = build_cpm(mv2_branch_2)
        return conv5_2_CPM, Mconv7_stage2, Mconv7_stage3

        l2s = []
        prev = None
        for stage_number in range(STAGE_NUM):
            if prev is not None:
                inputs = tf.concat([cancat_mv2, prev], axis=3)
            else:
                inputs = cancat_mv2

            kernel_size = 7
            lastest_channel_size = 128
            if stage_number == 0:
                kernel_size = 3
                lastest_channel_size = 512

            _ = slim.stack(inputs, inverted_bottleneck,
                           [
                               (2, out_channel_cpm(32), 0, kernel_size),
                               (up_channel_ratio(4), out_channel_cpm(32), 0, kernel_size),
                               (up_channel_ratio(4), out_channel_cpm(32), 0, kernel_size),
                           ], scope="stage_%d_mv2" % stage_number)

            _ = slim.stack(_, separable_conv,
                           [
                               (out_channel_ratio(lastest_channel_size), 1, 1),
                               (N_KPOINTS, 1, 1)
                           ], scope="stage_%d_mv1" % stage_number)

            prev = _
            cpm_out = upsample(_, 1, "stage_%d_out" % stage_number)
            l2s.append(cpm_out)

    return cpm_out, l2s

l2s = []


def hourglass_module(inp, stage_nums):
    if stage_nums > 0:
        down_sample = max_pool(inp, 2, 2, 2, 2, name="hourglass_downsample_%d" % stage_nums)

        block_front = slim.stack(down_sample, inverted_bottleneck,
                                 [
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                 ], scope="hourglass_front_%d" % stage_nums)
        stage_nums -= 1
        block_mid = hourglass_module(block_front, stage_nums)
        block_back = inverted_bottleneck(
            block_mid, up_channel_ratio(6), N_KPOINTS,
            0, 3, scope="hourglass_back_%d" % stage_nums)

        up_sample = upsample(block_back, 2, "hourglass_upsample_%d" % stage_nums)
        #input(up_sample.get_shape().as_list())
        # jump layer
        branch_jump = slim.stack(inp, inverted_bottleneck,
                                 [
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), N_KPOINTS, 0, 3),
                                 ], scope="hourglass_branch_jump_%d" % stage_nums)

        curr_hg_out = tf.add(up_sample, branch_jump, name="hourglass_out_%d" % stage_nums)
        # mid supervise
        #curr_upsample = upsample(curr_hg_out, 2, "curr_upsample_%d" % stage_nums)
        #input(curr_hg_out.get_shape().as_list())
        l2s.append(curr_hg_out)

        return curr_hg_out

    _ = inverted_bottleneck(
        inp, up_channel_ratio(6), out_channel_ratio(24),
        0, 3, scope="hourglass_mid_%d" % stage_nums
    )
    return _


def build_network3(input, trainable):
    is_trainable(trainable)

    net = convb(input, 3, 3, out_channel_ratio(16), 2, name="Conv2d_0")

    # 128, 112
    net = slim.stack(net, inverted_bottleneck,
                     [
                         (1, out_channel_ratio(16), 0, 3),
                         (1, out_channel_ratio(16), 0, 3)
                     ], scope="Conv2d_1")

    # 64, 56
    net = slim.stack(net, inverted_bottleneck,
                     [
                         (up_channel_ratio(6), out_channel_ratio(24), 1, 3),
                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                     ], scope="Conv2d_2")

    net_h_w = int(net.shape[1])
    # build network recursively
    hg_out = hourglass_module(net, 3)
    crap0 = l2s[0]
    crap1 = l2s[1]
    crap2 = l2s[2]

    crap0 = upsample(crap0, 2, "output_upsample_1")
    crap2 = max_pool(crap2, 2, 2, 2, 2, name="output_upsample_3")
    return crap0, crap1, crap2
