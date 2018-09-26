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
        l2s = []

        #pool_center_lower = layers.avg_pool2d(center_map, 9, 8, padding='SAME')
        conv1_1 = layers.conv2d(input, 64, 3, 1, activation_fn=None, scope='conv1_1')
        conv1_1 = tf.nn.relu(conv1_1)
        conv1_2 = layers.conv2d(conv1_1, 64, 3, 1, activation_fn=None, scope='conv1_2')
        conv1_2 = tf.nn.relu(conv1_2)
        pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
        conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1, activation_fn=None, scope='conv2_1')
        conv2_1 = tf.nn.relu(conv2_1)
        conv2_2 = layers.conv2d(conv2_1, 128, 3, 1, activation_fn=None, scope='conv2_2')
        conv2_2 = tf.nn.relu(conv2_2)
        pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
        conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1, activation_fn=None, scope='conv3_1')
        conv3_1 = tf.nn.relu(conv3_1)
        conv3_2 = layers.conv2d(conv3_1, 256, 3, 1, activation_fn=None, scope='conv3_2')
        conv3_2 = tf.nn.relu(conv3_2)
        conv3_3 = layers.conv2d(conv3_2, 256, 3, 1, activation_fn=None, scope='conv3_3')
        conv3_3 = tf.nn.relu(conv3_3)
        conv3_4 = layers.conv2d(conv3_3, 256, 3, 1, activation_fn=None, scope='conv3_4')
        conv3_4 = tf.nn.relu(conv3_4)
        #pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
        conv4_1 = layers.conv2d(conv3_4, 512, 3, 1, activation_fn=None, scope='conv4_1')
        conv4_1 = tf.nn.relu(conv4_1)
        conv4_2 = layers.conv2d(conv4_1, 512, 3, 1, activation_fn=None, scope='conv4_2')
        conv4_2 = tf.nn.relu(conv4_2)
        conv4_3_CPM = layers.conv2d(conv4_2, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')
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
        return conv5_2_CPM, Mconv7_stage2, Mconv7_stage3

        return cpm_out, l2s
