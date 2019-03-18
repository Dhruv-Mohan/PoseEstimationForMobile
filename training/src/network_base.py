# -*- coding: utf-8 -*-
# @Time    : 18-4-24 5:48 PM
# @Author  : edvard_hua@live.com
# @FileName: network_base.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.contrib.slim as slim

_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_trainable = True
import tensorflow.contrib.layers as layers

def is_trainable(trainable=True):
    global _trainable
    _trainable = trainable



def max_pool(inputs, k_h, k_w, s_h, s_w, name, padding="SAME"):
    return tf.nn.max_pool(inputs,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)


def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(32), int(32)],
                                    name=name)


def upsample2(inputs, factor, name):
   return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor], name=name)


def separable_conv(input, c_o, k_s, stride, scope):

        with slim.arg_scope([slim.batch_norm],
                            decay=0.9,
                            fused=True,
                            is_training=_trainable,
                            activation_fn=tf.nn.relu):
            output = slim.separable_convolution2d(input,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  trainable=_trainable,
                                                  depth_multiplier=1.0,
                                                  kernel_size=[k_s, k_s],
                                                  weights_initializer=_init_xavier,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  scope=scope + '_depthwise')

            output = slim.convolution2d(output,
                                        c_o,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        trainable=_trainable,
                                        weights_regularizer=None,
                                        scope=scope + '_pointwise')

        #output = layers.separable_conv2d(input, c_o, kernel_size=1, depth_multiplier=1, scope=scope + '_depthwise')

        return output


def inverted_bottleneck(inputs, up_channel_rate, channels, subsample, k_s=3, scope=""):
    with tf.variable_scope("inverted_bottleneck_%s" % scope):
        with slim.arg_scope([slim.batch_norm],
                            decay=0.9,
                            fused=True,
                            is_training=_trainable,
                            activation_fn=tf.nn.relu6):
            stride = 2 if subsample else 1

            output = slim.convolution2d(inputs,
                                        up_channel_rate * inputs.get_shape().as_list()[-1],
                                        stride=1,
                                        kernel_size=[1, 1],
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=None,
                                        scope=scope + '_up_pointwise',
                                        trainable=_trainable)

            output = slim.separable_convolution2d(output,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  depth_multiplier=1.0,
                                                  kernel_size=k_s,
                                                  weights_initializer=_init_xavier,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  padding="SAME",
                                                  scope=scope + '_depthwise',
                                                  trainable=_trainable)

            output = slim.convolution2d(output,
                                        channels,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=None,
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=None,
                                        scope=scope + '_pointwise',
                                        trainable=_trainable)
            if inputs.get_shape().as_list()[-1] == channels:
                output = tf.add(inputs, output)

    return output


def convb(input, k_h, k_w, c_o, stride, name, relu=True):
    with slim.arg_scope([slim.batch_norm], decay=0.9, fused=True, is_training=_trainable):
        output = slim.convolution2d(
            inputs=input,
            num_outputs=c_o,
            kernel_size=[k_h, k_w],
            stride=stride,
            normalizer_fn=slim.batch_norm,
            weights_regularizer=_l2_regularizer_00004,
            weights_initializer=_init_xavier,
            biases_initializer=_init_zero,
            activation_fn=tf.nn.relu if relu else None,
            scope=name,
            trainable=_trainable)
    return output


def base_HG_block(input, channels=96, scope_prefix=""):
   branch_1 = inverted_bottleneck(input, 2, channels, 0, 3, scope= scope_prefix+'branch1')
   #branch_2 = inverted_bottleneck(branch_1, 2, channels/2, 0, 3, scope= scope_prefix+'branch2')
   #branch_3 = inverted_bottleneck(branch_2, 2, channels/4, 0 ,3, scope= scope_prefix+'branch3')

   #cat_1 = tf.concat(axis=3, values=[branch_1, input])

   residual = tf.add(x=branch_1, y=input)
   residual = tf.nn.relu6(residual)
   return tf.scalar_mul(0.01, residual)

def hourglass_block(input, scope_prefix=""):
   sub_stage_1 = base_HG_block(input, scope_prefix=scope_prefix+ '1') #64x64
   #side_stage_1 = base_HG_block(sub_stage_1, scope_prefix=scope_prefix+ '1s')
   side_stage_1 = sub_stage_1
   pool1 = tf.nn.max_pool(sub_stage_1,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding="SAME")

   sub_stage_2 = base_HG_block(pool1, scope_prefix=scope_prefix+ '2') #32x32
   #side_stage_2 = base_HG_block(sub_stage_2, scope_prefix=scope_prefix+ '2s')
   side_stage_2 = sub_stage_2
   pool2 = tf.nn.max_pool(sub_stage_2,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding="SAME")

   sub_stage_3 = base_HG_block(pool2, scope_prefix=scope_prefix+ '3')  # 16x16
   #side_stage_3 = base_HG_block(sub_stage_3, scope_prefix=scope_prefix+ '3s')
   side_stage_3 = sub_stage_3
   pool3 = tf.nn.max_pool(sub_stage_3,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding="SAME")

   sub_stage_4 = base_HG_block(pool3, scope_prefix=scope_prefix+ '4')  # 8x8
   #side_stage_4 = base_HG_block(sub_stage_4, scope_prefix=scope_prefix+ '4s')
   side_stage_4 = sub_stage_4
   pool4 = tf.nn.max_pool(sub_stage_4,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding="SAME")

   sub_stage_5a = base_HG_block(pool4, scope_prefix=scope_prefix+ '5a')         # 4x4
   #sub_stage_5b = base_HG_block(sub_stage_5a, scope_prefix=scope_prefix+ '5b')  # 4x4
   #sub_stage_5c = base_HG_block(sub_stage_5b, scope_prefix=scope_prefix+ '5c')  # 4x4

   up_stage_1 = upsample2(sub_stage_5a, 2, scope_prefix+'up1') # 8x8
   res_stage_1 = tf.add(x=up_stage_1, y=side_stage_4)
   sub_stage_6 = base_HG_block(res_stage_1, scope_prefix=scope_prefix+ '6')

   up_stage_2 = upsample2(sub_stage_6, 2, scope_prefix+'up2') # 16x16
   res_stage_2 = tf.add(x=up_stage_2, y=side_stage_3)
   sub_stage_7 = base_HG_block(res_stage_2, scope_prefix=scope_prefix+ '7')

   up_stage_3 = upsample2(sub_stage_7, 2, scope_prefix+'up3') # 32x32
   res_stage_3 = tf.add(x=up_stage_3, y=side_stage_2, )
   sub_stage_8 = base_HG_block(res_stage_3, scope_prefix=scope_prefix+ '8')

   up_stage_4 = upsample2(sub_stage_8, 2, scope_prefix+'up4') # 64x64
   res_stage_4 = tf.add(x=up_stage_4, y=side_stage_1)
   sub_stage_9 = base_HG_block(res_stage_4, scope_prefix=scope_prefix+ '9')

   return sub_stage_9




def base_HG_block(input, channels=64, scope_prefix="", mobile=True):
   if mobile:
       branch_1 = inverted_bottleneck(input, 2, channels, 0, 3, scope= scope_prefix+'branch1')
       #branch_2 = inverted_bottleneck(branch_1, 2, channels/2, 0, 3, scope= scope_prefix+'branch2')
       #branch_3 = inverted_bottleneck(branch_2, 2, channels/4, 0 ,3, scope= scope_prefix+'branch3')

       #cat_1 = tf.concat(axis=3, values=[branch_1, input])

       residual = tf.add(x=branch_1, y=input)
       residual = tf.nn.relu6(residual)

   else:
       branch_1 = convb(input, 3, 3,  256/2, 1, relu=False, name= scope_prefix + 'branch1')
       branch_2 = convb(branch_1, 3, 3, 256 / 4, 1, relu=False, name=scope_prefix + 'branch2')
       branch_3 = convb(branch_2, 3, 3, 256 / 4, 1, relu=False, name=scope_prefix + 'branch3')

       cat_1 = tf.concat(axis=3, values=[branch_1, branch_2, branch_3])

       residual = tf.add(x=cat_1, y=input)
       residual = tf.nn.relu(residual)

   return tf.scalar_mul(0.001, residual)

def hourglass_block(input, scope_prefix="", mobile=True):
   if mobile:
       sub_stage_1 = base_HG_block(input, scope_prefix=scope_prefix+ '1') #64x64
       #side_stage_1 = base_HG_block(sub_stage_1, scope_prefix=scope_prefix+ '1s')
       side_stage_1 = sub_stage_1
       pool1 = tf.nn.max_pool(sub_stage_1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")

       sub_stage_2 = base_HG_block(pool1, scope_prefix=scope_prefix+ '2') #32x32
       #side_stage_2 = base_HG_block(sub_stage_2, scope_prefix=scope_prefix+ '2s')
       side_stage_2 = sub_stage_2
       pool2 = tf.nn.max_pool(sub_stage_2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")

       sub_stage_3 = base_HG_block(pool2, scope_prefix=scope_prefix+ '3')  # 16x16
       #side_stage_3 = base_HG_block(sub_stage_3, scope_prefix=scope_prefix+ '3s')
       side_stage_3 = sub_stage_3
       pool3 = tf.nn.max_pool(sub_stage_3,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

       sub_stage_4 = base_HG_block(pool3, scope_prefix=scope_prefix+ '4')  # 8x8
       #side_stage_4 = base_HG_block(sub_stage_4, scope_prefix=scope_prefix+ '4s')
       side_stage_4 = sub_stage_4
       pool4 = tf.nn.max_pool(sub_stage_4,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

       sub_stage_5a = base_HG_block(pool4, scope_prefix=scope_prefix+ '5a')         # 4x4
       #sub_stage_5b = base_HG_block(sub_stage_5a, scope_prefix=scope_prefix+ '5b')  # 4x4
       #sub_stage_5c = base_HG_block(sub_stage_5b, scope_prefix=scope_prefix+ '5c')  # 4x4

       up_stage_1 = upsample2(sub_stage_5a, 2, scope_prefix+'up1') # 8x8
       res_stage_1 = tf.add(x=up_stage_1, y=side_stage_4)
       sub_stage_6 = base_HG_block(res_stage_1, scope_prefix=scope_prefix+ '6')

       up_stage_2 = upsample2(sub_stage_6, 2, scope_prefix+'up2') # 16x16
       res_stage_2 = tf.add(x=up_stage_2, y=side_stage_3)
       sub_stage_7 = base_HG_block(res_stage_2, scope_prefix=scope_prefix+ '7')

       up_stage_3 = upsample2(sub_stage_7, 2, scope_prefix+'up3') # 32x32
       res_stage_3 = tf.add(x=up_stage_3, y=side_stage_2, )
       sub_stage_8 = base_HG_block(res_stage_3, scope_prefix=scope_prefix+ '8')

       up_stage_4 = upsample2(sub_stage_8, 2, scope_prefix+'up4') # 64x64
       res_stage_4 = tf.add(x=up_stage_4, y=side_stage_1)
       sub_stage_9 = base_HG_block(res_stage_4, scope_prefix=scope_prefix+ '9')

   else:
       sub_stage_1 = base_HG_block(input, scope_prefix=scope_prefix + '1', mobile=False)  # 64x64
       side_stage_1 = base_HG_block(sub_stage_1, scope_prefix=scope_prefix+ '1s',  mobile=False)
       pool1 = tf.nn.max_pool(sub_stage_1,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

       sub_stage_2 = base_HG_block(pool1, scope_prefix=scope_prefix + '2',  mobile=False)  # 32x32
       side_stage_2 = base_HG_block(sub_stage_2, scope_prefix=scope_prefix+ '2s', mobile=False)
       pool2 = tf.nn.max_pool(sub_stage_2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

       sub_stage_3 = base_HG_block(pool2, scope_prefix=scope_prefix + '3', mobile=False)  # 16x16
       side_stage_3 = base_HG_block(sub_stage_3, scope_prefix=scope_prefix+ '3s', mobile=False)
       pool3 = tf.nn.max_pool(sub_stage_3,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

       sub_stage_4 = base_HG_block(pool3, scope_prefix=scope_prefix + '4', mobile=False)  # 8x8
       side_stage_4 = base_HG_block(sub_stage_4, scope_prefix=scope_prefix+ '4s', mobile=False)
       pool4 = tf.nn.max_pool(sub_stage_4,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

       sub_stage_5a = base_HG_block(pool4, scope_prefix=scope_prefix + '5a', mobile=False)  # 4x4
       sub_stage_5b = base_HG_block(sub_stage_5a, scope_prefix=scope_prefix+ '5b', mobile=False)  # 4x4
       sub_stage_5c = base_HG_block(sub_stage_5b, scope_prefix=scope_prefix+ '5c', mobile=False)  # 4x4

       up_stage_1 = upsample2(sub_stage_5c, 2, scope_prefix + 'up1')  # 8x8
       res_stage_1 = tf.add(x=up_stage_1, y=side_stage_4)
       sub_stage_6 = base_HG_block(res_stage_1, scope_prefix=scope_prefix + '6', mobile=False)

       up_stage_2 = upsample2(sub_stage_6, 2, scope_prefix + 'up2')  # 16x16
       res_stage_2 = tf.add(x=up_stage_2, y=side_stage_3)
       sub_stage_7 = base_HG_block(res_stage_2, scope_prefix=scope_prefix + '7', mobile=False)

       up_stage_3 = upsample2(sub_stage_7, 2, scope_prefix + 'up3')  # 32x32
       res_stage_3 = tf.add(x=up_stage_3, y=side_stage_2, )
       sub_stage_8 = base_HG_block(res_stage_3, scope_prefix=scope_prefix + '8', mobile=False)

       up_stage_4 = upsample2(sub_stage_8, 2, scope_prefix + 'up4')  # 64x64
       res_stage_4 = tf.add(x=up_stage_4, y=side_stage_1)
       sub_stage_9 = base_HG_block(res_stage_4, scope_prefix=scope_prefix + '9', mobile=False)

   return sub_stage_9