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
import os
import time
import numpy as np
import configparser
import dataset
import cv2
from datetime import datetime

from dataset import get_train_dataset_pipeline
from networks import get_network
from dataset_prepare import CocoPose
from dataset_augment import set_network_input_wh, set_network_scale

import torch
import torch.nn.functional as F
import numbers, math
import numpy as np

_WIDTH_ = 256
_HEIGHT_ = 256
cpu = torch.device('cpu')
_POINTS_ = 99
def get_input(batchsize, epoch, is_train=True):
    if True:
        input_pipeline = get_train_dataset_pipeline(batch_size=batchsize, epoch=epoch, buffer_size=100)
    #else:
    #input_pipeline = get_valid_dataset_pipeline(batch_size=batchsize, epoch=epoch, buffer_size=100)
    iter = input_pipeline.make_one_shot_iterator()
    _ = iter.get_next()
    return _[0], _[1], _[2],_[3] , _[4], _[5],_[6] , _[7]

def get_locs_from_hmap(part_map_resized):
    return (np.unravel_index(part_map_resized.argmax(), part_map_resized.shape))

def get_loss_and_output(model, batchsize, input_image, input_heat1, input_heat2, input_heat3,input_heat1l, input_heat2l, input_heat3l, heat_mean, reuse_variables=None):
    losses = []

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        conv5_2_CPM, Mconv7_stage2, Mconv7_stage3, conv5_2_CPMl, Mconv7_stage2l, Mconv7_stage3l, output = get_network(model, input_image, True)
        #_, pred_heatmaps_all = get_network(model, input_image, False)

    ll0 = tf.nn.l2_loss(tf.concat(conv5_2_CPM, axis=0) - input_heat1, name='loss_heatmap_stage%d' % 0)


    ll1 = tf.nn.l2_loss(tf.concat(Mconv7_stage2, axis=0) - input_heat2, name='loss_heatmap_stage%d' % 1)


    ll2 = tf.nn.l2_loss(tf.concat(Mconv7_stage3, axis=0) - input_heat3, name='loss_heatmap_stage%d' % 2)

    ll0l = tf.nn.l2_loss(tf.concat(conv5_2_CPMl, axis=0) - input_heat1l, name='loss_heatmap_stage%d' % 0)


    ll1l = tf.nn.l2_loss(tf.concat(Mconv7_stage2l, axis=0) - input_heat2l, name='loss_heatmap_stage%d' % 1)


    ll2l = tf.nn.l2_loss(tf.concat(Mconv7_stage3l, axis=0) - input_heat3l, name='loss_heatmap_stage%d' % 2)

    losses.append(ll2)
    losses.append(ll1)
    losses.append(ll0)
    losses.append(ll2l)
    losses.append(ll1l)
    losses.append(ll0l)
    '''
    for idx, pred_heat in enumerate(pred_heatmaps_all):
        loss_l2 = tf.nn.l2_loss(tf.concat(pred_heat, axis=0) - input_heat, name='loss_heatmap_stage%d' % idx)
        losses.append(loss_l2)
        
    total_loss = tf.reduce_sum(losses) / batchsize
    total_loss_ll_heat = tf.reduce_sum(loss_l2) / batchsize
    return total_loss, total_loss_ll_heat, pred_heat

    '''
    total_loss = tf.reduce_sum(losses) / batchsize
    total_loss_ll_heat = tf.reduce_sum(ll2) / batchsize
    return total_loss, total_loss_ll_heat, output

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
            else:
                print(var_name)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def average_gradients(tower_grads):
    """
    Get gradients of all variables.
    :param tower_grads:
    :return:
    """
    average_grads = []

    # get variable and gradients in differents gpus
    for grad_and_vars in zip(*tower_grads):
        # calculate the average gradient of each gpu
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def find_tensor_peak_batch(heatmap, radius, downsample, threshold=0.000001):
    heatmap = heatmap[0, :, :, :]
    heatmap = np.transpose(heatmap, [2, 0, 1])
    heatmap = torch.from_numpy(heatmap)
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()

    def normalize(x, L):
        return -1. + 2. * x.data / (L - 1)

    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    # affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    # theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:, 0, 0] = (boxes[2] - boxes[0]) / 2
    affine_parameter[:, 0, 2] = (boxes[2] + boxes[0]) / 2
    affine_parameter[:, 1, 1] = (boxes[3] - boxes[1]) / 2
    affine_parameter[:, 1, 2] = (boxes[3] + boxes[1]) / 2
    # extract the sub-region heatmap
    theta = affine_parameter.to(heatmap.device)
    grid_size = torch.Size([num_pts, 1, radius * 2 + 1, radius * 2 + 1])
    grid = F.affine_grid(theta, grid_size)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

    X = torch.arange(-radius, radius + 1).to(heatmap).view(1, 1, radius * 2 + 1)
    Y = torch.arange(-radius, radius + 1).to(heatmap).view(1, radius * 2 + 1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts, -1), 1)
    x = torch.sum((sub_feature * X).view(num_pts, -1), 1) / sum_region + index_w
    y = torch.sum((sub_feature * Y).view(num_pts, -1), 1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y], 1)

def process_pt(pt):
    return pt[0]*16 +pt[1]

def main(argv=None):
    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config_file = "experiments/mv2_cpm.cfg"
    if len(argv) != 1:
        config_file = argv[1]
    config.read(config_file)
    for _ in config.options("Train"):
        params[_] = eval(config.get("Train", _))

    os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']

    gpus_index = params['visible_devices'].split(",")
    params['gpus'] = len(gpus_index)

    if not os.path.exists(params['modelpath']):
        os.makedirs(params['modelpath'])
    if not os.path.exists(params['logpath']):
        os.makedirs(params['logpath'])

    dataset.set_config(params)
    set_network_input_wh(params['input_width'], params['input_height'])
    set_network_scale(params['scale'])

    training_name = '{}_batch-{}_lr-{}_gpus-{}_{}x{}_{}'.format(
        params['model'],
        params['batchsize'],
        params['lr'],
        params['gpus'],
        params['input_width'], params['input_height'],
        config_file.replace("/", "-").replace(".cfg", "")
    )

    with tf.Graph().as_default(), tf.device("/cpu:0"):
        input_image, input_heat1, input_heat2, input_heat3, input_heat1l, input_heat2l, input_heat3l, heat_mean  = get_input(params['batchsize'], params['max_epoch'], is_train=True)
        #valid_input_image, valid_input_heat = get_input(params['batchsize'], params['max_epoch'], is_train=False)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(float(params['lr']) *20, global_step,
                                                   decay_steps=15000, decay_rate=float(params['decay_rate']), staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        #tower_grads = []
        reuse_variable = False

        # multiple gpus
        for i in range(params['gpus']):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("GPU_%d" % i):
                    loss, last_heat_loss, pred_heat = get_loss_and_output(params['model'], params['batchsize'], input_image, input_heat1, input_heat2, input_heat3, input_heat1l, input_heat2l, input_heat3l, heat_mean, reuse_variable)
                    reuse_variable = True
                    grads = opt.compute_gradients(loss)
                    #tower_grads.append(grads)

        #grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        MOVING_AVERAGE_DECAY = 0.99
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(max_to_keep=100)

        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("loss_lastlayer_heat", last_heat_loss)
        summary_merge_op = tf.summary.merge_all()

        pred_result_image = tf.placeholder(tf.float32, shape=[params['batchsize'], 480, 640, 3])
        pred_result__summary = tf.summary.image("pred_result_image", pred_result_image, params['batchsize'])

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            latest_ckpt = tf.train.latest_checkpoint(
                'model/mv2_cpm_batch-64_lr-0.0005_gpus-1_256x256_experiments-mv2_cpm')
            if latest_ckpt:
                print('Ckpt_found')
                #print(latest_ckpt)
                optimistic_restore(sess, latest_ckpt)
                #saver.restore(sess, latest_ckpt)
                #input(latest_ckpt)
                #saver.restore(sess, 'model/mv2_cpm_batch-64_lr-0.01_gpus-1_256x256_experiments-mv2_cpm/model-1000')

            else:
                print('CKPT not found')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.summary.FileWriter(os.path.join(params['logpath'], training_name), sess.graph)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize'] * params['gpus'])
            print("Start training...")

            for step in range(total_step_num):
                start_time = time.time()

                _, loss_value, lh_loss, in_image, in_heat, p_heat = sess.run(
                    [train_op, loss, last_heat_loss, input_image, input_heat3, pred_heat]
                )
                #print(in_image.shape)
                '''
                in_image, in_heat, p_heat = sess.run(
                    [input_image, input_heat, pred_heat]
                )
                '''
                duration = time.time() - start_time
                print("Duration : {}\t Loss : {}".format(duration, loss_value))

                in_image = np.asarray(in_image)
                in_image = in_image[0, :, :, :]

                heat = np.asarray(p_heat)
                #locs = find_tensor_peak_batch(heat, 4, 8)
                #locs = locs.to(cpu).numpy()
                # input(locs.shape)
                heat = heat[0, :, :, :]
                in_heat = np.asarray(in_heat)
                in_heat = in_heat[0, :, :, :]
                coords = []
                gt_coords = []
                sum_heatmap = np.zeros([_WIDTH_, _WIDTH_])
                gt_sum_heatmap = np.zeros([_WIDTH_, _WIDTH_])
                for i in range(_POINTS_ -1):
                    single_heatmap = cv2.resize(heat[:,:, i], (_WIDTH_,_WIDTH_), cv2.INTER_LANCZOS4)
                    sum_heatmap += single_heatmap
                    pt = get_locs_from_hmap(single_heatmap)
                    coords.append(pt)
                '''
                for i in range(0,180,2):
                    single_heatmap_x = cv2.resize(heat[:, :, i], (16, 16), cv2.INTER_LANCZOS4)
                    single_heatmap_y = cv2.resize(heat[:, :, i + 1], (16, 16), cv2.INTER_LANCZOS4)
                    ptx = get_locs_from_hmap(single_heatmap_x)

                    pty = get_locs_from_hmap(single_heatmap_y)
                    # input(pty)
                    ptx, pty = process_pt(ptx), process_pt(pty)
                    # input(pty)
                    # ptx *= 8
                    # pty *= 8
                    # input(ptx)
                    coords.append([pty, ptx])
                    #single_in_heat = cv2.resize(in_heat[:, :, i], (256, 256), 0)
                
                    # cv2.circle(in_image, (int(locs[i][0]), int(locs[i][1])), 3, (255, 0, 255), 1)
                    # input(pt)
                '''

                '''
                for pt in coords:
                    cv2.circle(in_image, (int(pt[1])*8, int(pt[0])*8), 3, (255, 0, 0), 1)
                
                for gtpt in gt_coords:
                    cv2.circle(in_image, (int(gtpt[1]), int(gtpt[0])), 3, (0, 0, 255), 1)
                '''

                for pt in coords:
                    cv2.circle(in_image, (int(pt[1]), int(pt[0]) ), 3, (255, 0, 255), 1)

                cv2.imshow('crap', in_image/255)


                sum_heatmap -= np.min(sum_heatmap)
                sum_heatmap /= np.max(sum_heatmap)
                sum_heatmap = np.expand_dims(sum_heatmap, -1)
                #gt_sum_heatmap -= np.min(gt_sum_heatmap)
                #gt_sum_heatmap /= np.max(gt_sum_heatmap)
                #gt_sum_heatmap = np.expand_dims(gt_sum_heatmap, -1)
                cv2.imshow('sum_heatmap', sum_heatmap)
                #cv2.imshow('gt_sum_heatmap', gt_sum_heatmap)
                cv2.imwrite('/home/dhruv/Projects/Datasets/in_image.jpg', in_image)

                #cv2.imwrite('/home/dhruv/Projects/Datasets/sum_heatmap.jpg', sum_heatmap * 255)
                #cv2.imwrite('/home/dhruv/Projects/Datasets/gt_sum_heatmap.jpg', gt_sum_heatmap * 255)

                cv2.waitKey(1)

                if step % params['per_saved_model_step'] == 0:
                    checkpoint_path = os.path.join(params['modelpath'], training_name, 'model')
                    saver.save(sess, checkpoint_path, global_step=step)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()