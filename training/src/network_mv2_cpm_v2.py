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

out_channel_ratio = lambda d: max(int(d * 0.75), 8)
up_channel_ratio = lambda d: int(d * 1.)
out_channel_cpm = lambda d: max(int(d * 0.75), 8)


def build_network(input, trainable):
    is_trainable(trainable)

    net = convb(input, 3, 3, out_channel_ratio(32), 2, name="Conv2d_0")

    with tf.variable_scope('MobilenetV2'):

        # 128, 112 192
        mv2_branch_0 = slim.stack(net, inverted_bottleneck,
                                  [
                                      (1, out_channel_ratio(16), 0, 3),
                                      (1, out_channel_ratio(16), 0, 3)
                                  ], scope="MobilenetV2_part_0")

        # 64, 56 96
        mv2_branch_1 = slim.stack(mv2_branch_0, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(24), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                  ], scope="MobilenetV2_part_1")

        # 32, 28 48
        mv2_branch_2 = slim.stack(mv2_branch_1, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(32), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                  ], scope="MobilenetV2_part_2")

        # 16, 14 24
        mv2_branch_3 = slim.stack(mv2_branch_2, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(64), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                  ], scope="MobilenetV2_part_3")

        # 8, 7 12
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
                max_pool(mv2_branch_0, 8, 8, 8, 8, name="mv2_0_max_pool"),
                max_pool(mv2_branch_1, 4, 4, 4, 4, name="mv2_1_max_pool"),
                max_pool(mv2_branch_2, 2, 2, 2, 2, name="mv2_1_max_pool"),
                mv2_branch_3
            ]
            , axis=3)

    conv4_4 = layers.conv2d(cancat_mv2, 256, 3, 1, activation_fn=None, scope='conv4_4')
    conv4_4 = tf.nn.relu(conv4_4)
    conv5_1 = layers.conv2d(conv4_4, 128, 3, 1, activation_fn=None, scope='conv5_1')
    conv5_1 = tf.nn.relu(conv5_1)

    with tf.variable_scope("Convolutional_Pose_Machine"):
        l2s = []
        prev = None
        for stage_number in range(STAGE_NUM):
            if prev is not None:
                inputs = tf.concat([conv5_1, prev], axis=3)
            else:
                inputs = conv5_1

            if stage_number is 0:
                s1_1 = layers.conv2d(inputs, 128, 3, 1, activation_fn=None, scope='s1_1')
                s1_2 = layers.conv2d(s1_1, 128, 3, 1, activation_fn=None, scope='s1_2')
                s1_3 = layers.conv2d(s1_2, 128, 3, 1, activation_fn=None, scope='s1_3')
                s1_4 = layers.conv2d(s1_3, 128, 3, 1, activation_fn=None, scope='s1_4')
                _ = slim.stack(s1_4, separable_conv,
                               [
                                   (out_channel_ratio(512), 1, 1),
                                   (N_KPOINTS, 1, 1)
                               ], scope="stage_%d_mv1" % stage_number)

                prev = _

            else:
                sx_1 = layers.conv2d(inputs, 128, 7, 1, activation_fn=None, scope='sx_1')
                sx_2 = layers.conv2d(sx_1, 128, 7, 1, activation_fn=None, scope='sx_2')
                sx_3 = layers.conv2d(sx_2, 128, 7, 1, activation_fn=None, scope='sx_3')
                sx_4 = layers.conv2d(sx_3, 128, 3, 1, activation_fn=None, scope='sx_4')
                sx_5 = layers.conv2d(sx_4, 128, 3, 1, activation_fn=None, scope='sx_5')
                sx_6 = layers.conv2d(sx_5, 128, 3, 1, activation_fn=None, scope='sx_6')
                sx_7 = layers.conv2d(sx_6, 128, 3, 1, activation_fn=None, scope='sx_7')
                _ = slim.stack(sx_7, separable_conv,
                               [
                                   (out_channel_ratio(128), 1, 1),
                               ], scope="stage_%d_mv1" % stage_number)

                cpm_out = separable_conv(_, N_KPOINTS, 1, 1, scope="stage_%d_out" % stage_number)
                prev = cpm_out
                l2s.append(cpm_out)

    return cpm_out, l2s
