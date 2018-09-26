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


cpu = torch.device('cpu')

def find_tensor_peak_batch(heatmap, radius, downsample, threshold=0.000001):
    heatmap = heatmap[0,:,:,:]
    heatmap = np.transpose(heatmap, [2, 0, 1])
    heatmap =  torch.from_numpy(heatmap)
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

def get_locs_from_hmap(part_map_resized):
    return(np.unravel_index(part_map_resized.argmax(), part_map_resized.shape))

def get_input(batchsize, epoch, is_train=True):
    input_pipeline = get_train_dataset_pipeline(batch_size=batchsize, epoch=epoch, buffer_size=100)
    iter = input_pipeline.make_one_shot_iterator()
    _ = iter.get_next()
    return _[0], _[1]


def get_loss_and_output(model, batchsize, input_image, input_heat, reuse_variables=None):
    losses = []

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        conv5_2_CPM, Mconv7_stage2, Mconv7_stage3 = get_network(model, input_image, False)

    ll0 = tf.nn.l2_loss(tf.concat(conv5_2_CPM, axis=0) - input_heat, name='loss_heatmap_stage%d' % 0)

    
    ll1 = tf.nn.l2_loss(tf.concat(Mconv7_stage2, axis=0) - input_heat, name='loss_heatmap_stage%d' % 1)

    
    ll2 = tf.nn.l2_loss(tf.concat(Mconv7_stage3, axis=0) - input_heat, name='loss_heatmap_stage%d' % 2)

    losses.append(ll2)
    losses.append(ll1)
    losses.append(ll0)
    '''
    for idx, pred_heat in enumerate(pred_heatmaps_all):
        loss_l2 = tf.nn.l2_loss(tf.concat(pred_heat, axis=0) - input_heat, name='loss_heatmap_stage%d' % idx)
        losses.append(loss_l2)
    '''
    total_loss = tf.reduce_sum(losses) / batchsize
    total_loss_ll_heat = tf.reduce_sum(ll2) / batchsize
    return total_loss, total_loss_ll_heat, Mconv7_stage3


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

def draw_pts(img, heat):
     for i in range(90):
         single_heat = heat[:,:,i]
         single_heat = cv2.resize(single_heat, (256,256), 0)
         pt = get_locs_from_hmap(single_heat)
         cv2.circle(img, (int(pt[1]), int(pt[0])), 5, (0.5,0.5,0))
     return img


def batch_draw_pts(images, heatmaps):
    return np.array([draw_pts(img, heat) for img, heat in zip(images, heatmaps)])

def draw_landmarks(images, heatmaps):
    return tf.py_func(batch_draw_pts,
               [images, heatmaps], [tf.float32])[0]

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

        #with tf.Graph().as_default():
        input_image, input_heat = get_input(params['batchsize'], params['max_epoch'], is_train=True)
        #valid_input_image, valid_input_heat = get_input(params['batchsize'], params['max_epoch'], is_train=False)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(float(params['lr']), global_step,
                                                   decay_steps=500, decay_rate=float(params['decay_rate']), staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        #tower_grads = []
        reuse_variable = False

        # multiple gpus
        for i in range(params['gpus']):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("GPU_%d" % i):
                    loss, last_heat_loss, pred_heat = get_loss_and_output(params['model'], params['batchsize'], input_image, input_heat, reuse_variable)
                    reuse_variable = True
                    grads = opt.compute_gradients(loss)
                    #tower_grads.append(grads)
                    #valid_loss, valid_last_heat_loss, valid_pred_heat = get_loss_and_output(params['model'], params['batchsize'],valid_input_image, valid_input_heat, reuse_variable)

        disp_image = draw_landmarks(input_image, pred_heat)
        disp_heat = tf.reduce_sum(pred_heat, -1)
        disp_heat = tf.expand_dims(disp_heat, -1)
        tf.summary.image('pred_pts', disp_image)
        tf.summary.image('disp_heat', disp_heat)
        #grads = average_gradients(tower_grads)
        '''
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)
        '''
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        MOVING_AVERAGE_DECAY = 0.99
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver()

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
            latest_ckpt = tf.train.latest_checkpoint('model/mv2_cpm_batch-64_lr-0.01_gpus-1_256x256_experiments-mv2_cpm')
            if latest_ckpt:
                print('Ckpt_found')
                print(latest_ckpt)
                saver.restore(sess, latest_ckpt)

            else:
                print('CKPT not found')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            summary_writer = tf.summary.FileWriter(os.path.join(params['logpath'], training_name), sess.graph)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize'] * params['gpus'])
            print("Start training...")
            for step in range(total_step_num):
                #input_image = np.zeros([256, 256, 3])
                sess.run([input_image, input_heat, pred_heat])
                start_time = time.time()
                #print("calling sess run...")

                _, loss_value, lh_loss, in_image, in_heat, p_heat = sess.run(
                    [train_op, loss, last_heat_loss, input_image, input_heat, pred_heat]
                )
                

                in_image, in_heat, p_heat = sess.run(
                    [input_image, input_heat, pred_heat]
                )
                
                duration = time.time() - start_time
                print("Duration : {}".format(duration))

                in_image = np.asarray(in_image)
                in_image= in_image[0,:,:,:] + 0.5

                heat = np.asarray(p_heat)
                locs = find_tensor_peak_batch(heat, 4, 8)
                locs = locs.to(cpu).numpy()
                #input(locs.shape)
                heat = heat[0,:,:,:]
                in_heat = np.asarray(in_heat)
                in_heat = in_heat[0,:,:,:]
                coords = []
                gt_coords = []
                sum_heatmap = np.zeros([256,256])
                gt_sum_heatmap = np.zeros([256,256])
                for i in range(90):
                    single_heatmap = cv2.resize(heat[:, :, i], (256, 256), 0)
                    single_in_heat = cv2.resize(in_heat[:, :, i], (256, 256), 0)
                    gt_sum_heatmap += single_in_heat
                    sum_heatmap +=single_heatmap
                    pt = get_locs_from_hmap(single_heatmap)
                    gtpt = get_locs_from_hmap(single_in_heat)
                    gt_coords.append(gtpt)
                    coords.append(pt)
                    #cv2.circle(in_image, (int(locs[i][0]), int(locs[i][1])), 3, (255, 0, 255), 1)
                    #input(pt)

                for pt in coords:
                    cv2.circle(in_image, (int(pt[1]), int(pt[0])), 3, (255, 255, 0), 1)

                
                for gtpt in gt_coords:
                    cv2.circle(in_image, (int(gtpt[1]), int(gtpt[0])), 3, (0, 0, 255), 1)


                cv2.imshow('crap', in_image)

                sum_heatmap -= np.min(sum_heatmap)
                sum_heatmap /= np.max(sum_heatmap)
                sum_heatmap = np.expand_dims(sum_heatmap, -1)
                gt_sum_heatmap -= np.min(gt_sum_heatmap)
                gt_sum_heatmap /= np.max(gt_sum_heatmap)
                gt_sum_heatmap = np.expand_dims(gt_sum_heatmap, -1)
                cv2.imshow('sum_heatmap', sum_heatmap)
                cv2.imshow('gt_sum_heatmap', gt_sum_heatmap)
                cv2.imwrite('/home/dhruv/Projects/Datasets/in_image.jpg', in_image * 255)
                cv2.imwrite('/home/dhruv/Projects/Datasets/sum_heatmap.jpg', sum_heatmap * 255)
                cv2.imwrite('/home/dhruv/Projects/Datasets/gt_sum_heatmap.jpg', gt_sum_heatmap * 255)
                cv2.waitKey(0)

                #print("out of sess run...")

                if step != 0 and step % params['per_update_tensorboard_step'] == 0:
                    print("in if loop...")
                    # False will speed up the training time.
                    if params['pred_image_on_tensorboard'] is True:
                        #print("out of sess run...")
                        #valid_loss_value, valid_lh_loss, valid_in_image, valid_in_heat, valid_p_heat = sess.run([valid_loss, valid_last_heat_loss, valid_input_image, valid_input_heat, valid_pred_heat])
                        #print("out of sess run...")
                        #result = []
                        #print("out of sess run...")
                        '''
                        for index in range(params['batchsize']):
                            r = CocoPose.display_image(
                                    valid_in_image[index,:,:,:],
                                    valid_in_heat[index,:,:,:],
                                    valid_p_heat[index,:,:,:],
                                    True
                                )
                            result.append(
                                r.astype(np.float32)
                            )
                        '''
                        #print("out of sess run...")

                        #summary_writer.add_summary(step)
                        #print("out of sess run...")
                    # print train info
                    num_examples_per_step = params['batchsize'] * params['gpus']
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / params['gpus']
                    format_str = ('%s: step %d, loss = %.2f, last_heat_loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, lh_loss, examples_per_sec, sec_per_batch))

                    # tensorboard visualization
                    merge_op = sess.run(summary_merge_op)
                    summary_writer.add_summary(merge_op, step)
                #print("out of if  loop...")
                # save model
                if (step + 1) % params['per_saved_model_step'] == 0:
                    print("inside saver loop...")
                    checkpoint_path = os.path.join(params['modelpath'], training_name, 'model')
                    saver.save(sess, checkpoint_path, global_step=global_step)
                #print("out of model save loop...")
            coord.request_stop()
            coord.join(threads)



if __name__ == '__main__':
    tf.app.run()