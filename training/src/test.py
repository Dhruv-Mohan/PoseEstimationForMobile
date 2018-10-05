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

import datetime
import time
from scipy.ndimage.filters import gaussian_filter
from utils.pointIO import *
from utils.draw import *
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

_GT_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/pts/'
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
cpu = torch.device('cpu')

def display_image():
    """
    display heatmap & origin image
    :return:
    """
    from dataset_prepare import CocoPose
    from pycocotools.coco import COCO
    from os.path import join
    from dataset import _parse_function

    BASE_PATH = ""

    # os.chdir("..")

    ANNO = COCO(
        join(BASE_PATH, "train_gm16k.json")
    )
    train_imgIds = ANNO.getImgIds()

    img, heat = _parse_function(train_imgIds[100], ANNO)

    CocoPose.display_image(img, heat, pred_heat=heat, as_numpy=False)

    from PIL import Image
    for _ in range(heat.shape[2]):
        data = CocoPose.display_image(img, heat, pred_heat=heat[:, :, _:(_ + 1)], as_numpy=True)
        im = Image.fromarray(data)
        im.save("test_heatmap/heat_%d.jpg" % _)


def saved_model_graph():
    """
    save the graph of model and check it in tensorboard
    :return:
    """

    from os.path import join
    from network_mv2_cpm_2 import build_network
    import tensorflow as tf
    import os


    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    input_node = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_HEIGHT, 3),
                                name='image')
    build_network(input_node, False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(
            join("tensorboard/test_graph/"),
            sess.graph
        )
        sess.run(tf.global_variables_initializer())


def metric_prefix(input_width, input_height):
    """
    output the calculation of you model
    :param input_width:
    :param input_height:
    :return:
    """
    import tensorflow as tf
    from networks import get_network
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    input_node = tf.placeholder(tf.float32, shape=(1, input_width, input_height, 3),
                                name='image')
    get_network("mv2_cpm_2", input_node, False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    run_meta = tf.RunMetadata()
    with tf.Session(config=config) as sess:
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("opts {:,} --- paras {:,}".format(flops.total_float_ops, params.total_parameters))
        sess.run(tf.global_variables_initializer())


def run_with_frozen_pb(img_path, input_w_h, frozen_graph, output_node_names):
    import tensorflow as tf
    import cv2
    import numpy as np
    import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from dataset_prepare import CocoPose
    with tf.device('/device:CPU:0'):
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())

        tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )

        graph = tf.get_default_graph()
        image = graph.get_tensor_by_name("image:0")
        output = graph.get_tensor_by_name("%s:0" % output_node_names)
        images = os.listdir('/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/val_set')
        total_l1e = []
        for ima in images:
            image_full_path = os.path.join('/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/val_set'
                                           , ima)
            image_output_path = os.path.join('/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/svrout',
                                             ima)
            gt_filename = os.path.splitext(ima)[0] +'.pts'
            gt_file_path = os.path.join(_GT_PATH_, gt_filename)
            gt_pts = get_pts(gt_file_path, 90)

            image_0 = cv2.imread(image_full_path)
            w, h, _ = image_0.shape
            gt_pts = scale_pts(gt_pts, image_0.shape)
            #input(gt_pts)
            image_ = cv2.resize(image_0, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
            image_ = image_.astype(np.float32)

            with tf.Session() as sess:
                #a = datetime.datetime.now()
                start_time = time.time()
                heatmaps = sess.run(output, feed_dict={image: [image_]})
                duration = time.time() - start_time
                print("Duration : {}".format(duration))
                heatmaps = np.squeeze(heatmaps)
                #c = datetime.datetime.now() - a
                #print(c.microseconds/1000)
                gt_sum_heatmap = np.zeros([512, 512])
                coords = []
                coords_32 = []
                coords_argmax = []
                heat = np.asarray(heatmaps)
                heat = np.expand_dims(heat, 0)
                locs, sub_f = find_tensor_peak_batch(heat, 4, 16)
                sub_f = sub_f.to(cpu).numpy()

                for i in range(90):
                    pt_32 = get_locs_from_hmap(heatmaps[:, :, i])
                    show_sub_f = np.expand_dims(sub_f[i, :, :], -1)
                    cv2.imshow('show_subf', show_sub_f)
                    pt_argmax = soft_argmax(heatmaps[:, :, i], pt_32)
                    print('subf ={}'.format(sub_f[i, :, :]))
                    cv2.waitKey(1)
                    coords_argmax.append(pt_argmax)
                    #pt_32[0] *= 16
                    #pt_32[0] *= 16
                    coords_32.append(pt_32)
                    single_heatmap = cv2.resize(heatmaps[:, :, i], (512, 512), 0)
                    gt_sum_heatmap += single_heatmap
                    pt = get_locs_from_hmap(single_heatmap)
                    coords.append(pt)
                image_ = cv2.resize(image_, (512,512), 0)
                #l1e = 0

                locs = locs.to(cpu).numpy()

                #input(locs.shape)
                image_, l1e = draw_pts(image_, gt_pts, coords_argmax, True)

                l1e = np.mean(l1e)
                print(l1e)
                total_l1e.append(l1e)
                #coords = cal_coord(heatmaps)
                '''
                for pt in coords:
                    cv2.circle(image_, (int(pt[1]), int(pt[0])),3,(255,255,0), 1)
                
                for pt in locs:
                    cv2.circle(image_, (int(pt[1])*16, int(pt[0])*16),3,(0,255,0), 1)
                '''
                for pt in coords_argmax:
                    cv2.circle(image_, (int(pt[0]), int(pt[1])),3,(255,255,0), 1)

                gt_sum_heatmap = np.expand_dims(gt_sum_heatmap, -1)
                gt_sum_heatmap= np.concatenate([gt_sum_heatmap, gt_sum_heatmap, gt_sum_heatmap], -1)
                cv2.imshow('image', image_/255)
                cv2.imshow('heatmap', gt_sum_heatmap)

                #gt_sum_heatmap = np.tile(gt_sum_heatmap, 3)
                cv2.imwrite(image_output_path, (image_))
                cv2.waitKey(0)
        total_l1e = np.asarray(total_l1e)
        total_l1e = np.mean(total_l1e)
        input('Total_l1e= {}'.format(total_l1e))
        CocoPose.display_image(
                # np.reshape(image_, [1, input_w_h, input_w_h, 3]),
                image_,
                None,
                heatmaps[0,:,:,:],
                False
        )
        # save each heatmaps to disk
        from PIL import Image
        for _ in range(heatmaps.shape[2]):
                data = CocoPose.display_image(image_, heatmaps[0,:,:,:], pred_heat=heatmaps[0, :, :, _:(_ + 1)], as_numpy=True)
                im = Image.fromarray(data)
                im.save("test/heat_%d.jpg" % _)

def get_locs_from_hmap(part_map_resized):
    return(np.unravel_index(part_map_resized.argmax(), part_map_resized.shape))

def soft_argmax(patch_t, coord):
    patch = cv2.getRectSubPix(patch_t, (9, 9), (coord[1], coord[0] ))
    test_patch = cv2.getRectSubPix(patch_t, (32, 32), (coord[1], coord[0]))
    show_patch = np.expand_dims(patch,-1)
    #test_show_patch = np.expand_dims(test_patch, -1)
    cv2.imshow('subrect', show_patch)
    #cv2.imshow('test subrect', test_show_patch*255)
    print('patch ={}'.format(patch))
    #input(heat_map.shape)
    patch_sum = np.sum(patch)
    #input('coord = {}'.format(coord))
    #input('patch_sum= {}'.format(patch_sum))
    x = np.linspace(-4, 4, 9)
    x = np.expand_dims(x, -1)
    y = np.transpose(x)
    #input('patch = {}'.format(patch))
    #input('patch*x = {}'.format(patch*y))
    #input('patch_sum = {}'.format(np.sum(patch*y)))
    x_pos = ((np.sum(patch*y) / patch_sum) + coord[1]) * 16 + 6.5
    #input('x_pos = {}'.format(x_pos))
    y_pos = ((np.sum(patch*x) / patch_sum) + coord[0]) * 16 + 6.5
    #input('y_pos = {}'.format(y_pos))
    return x_pos, y_pos


def scale_pts(pts, image_shape):
    w, h, _ = image_shape
    #input(image_shape)
    new_pts = np.ones([90, 2])
    for index, pt in enumerate(pts):
        new_pts[index, 0] = (pt[0] / h) * 512
        new_pts[index, 1] = (pt[1] / w) * 512
    return new_pts

def cal_coord(pred_heatmaps):
    heat_h, heat_w, n_kpoints = pred_heatmaps.shape
    scale_h = 8
    scale_w = 8
    coord = []
    for p_ind in range(n_kpoints):
            heat = pred_heatmaps[:, :, p_ind]
            heat = gaussian_filter(heat, sigma=2)
            ind = np.unravel_index(np.argmax(heat), heat.shape)
            coord_x = int((ind[1] + 1) * scale_w)
            coord_y = int((ind[0] + 1) *scale_h)
            coord.append((coord_x, coord_y))
    coords = coord
    return coords

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
    x_sum_region = sub_feature[0,:,:]
    X_test = torch.arange(-radius, radius + 1).to(heatmap).view(1, radius * 2 + 1)
    X_TEST_SUM_REGION = x_sum_region * X_test
    X_TEST_SUM = torch.sum(X_TEST_SUM_REGION)
    #input('X_TEST = {}'.format(x_sum_region * X_test))
    #input('X_TEST_SUM= {}'.format(X_TEST_SUM))
    x_sum_region = x_sum_region.to(cpu).numpy()
    X_test = X_test.to(cpu).numpy()
    num_x_sum_region =x_sum_region*X_test
    x_sum = np.sum(num_x_sum_region)
    input('X_TEST_NUMPY = {}'.format(num_x_sum_region))
    #input('INDEX_W= {}'.format(index_w[0]))
    #sum_region = sum_region.to(cpu).numpy()
    x = torch.sum((sub_feature * X).view(num_pts, -1), 1) / sum_region + index_w
    #input('X_TEST_SUM_numpy= {}'.format(x_sum))
    #input('X_ = {}'.format(X))
    #input('sub_feature*X_ = {}'.format((sub_feature * X)[0,:,:]))

    #input('sub_feature = {}'.format(sub_feature[0,:,:]))
    y = torch.sum((sub_feature * Y).view(num_pts, -1), 1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5

    y = y * downsample + downsample / 2.0 - 0.5
    #input('y_ = {}'.format(y[0]))
    return torch.stack([x, y], 1), sub_feature

if __name__ == '__main__':
    # saved_model_graph()
    #metric_prefix(192, 192)
    run_with_frozen_pb(
         "/home/dhruv/Projects/PersonalGit/PoseEstimationForMobile/training/Menpo51220/2.jpg",
         256,
         "./overfit_duo.pb",
         "Convolutional_Pose_Machine/Mconv7_stage3/separable_conv2d"
     )
    display_image()



def get_locs_from_heatmaps(heatmaps):
    coords = []
    for i in range(90):
        single_heatmap = cv2.resize(heatmaps[:, :, i], (256, 256), 0)
        pt = get_locs_from_hmap(single_heatmap)
        coords.append(pt)

    return coords

