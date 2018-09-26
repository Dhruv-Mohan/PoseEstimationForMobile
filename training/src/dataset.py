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

from dataset_augment import pose_random_scale, pose_rotation, pose_flip, pose_resize_shortestedge_random, \
    pose_crop_random, pose_to_img
from dataset_prepare import CocoMetadata
from os.path import join
from pycocotools.coco import COCO
import os
import numpy as np
import cv2
import pickle
import multiprocessing

BASE = "/root/hdd"
BASE_PATH = ""
#TRAIN_JSON = "train_gm16k.json"
#VALID_JSON = "val_gm16k.json"

TRAIN_ANNO = None
VALID_ANNO = None
CONFIG = None
_TRAIN_IMAGE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/single_train_images_256/'
_TRAIN_PICKLE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/mobile_train_256_32/'

#_VAL_IMAGE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/mobile_val_images_256/'
#_VAL_PICKLE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/mobile_val_256/'
def set_config(config):
    global CONFIG, BASE, BASE_PATH
    CONFIG = config
    BASE = CONFIG['imgpath']
    BASE_PATH = CONFIG['datapath']


def _parse_function(imgId, is_train, ann=None):
    imgId = imgId.decode()
    pickle_name = os.path.splitext(imgId)[0] + '.pickle'
    if is_train == True:
        image_path = join(_TRAIN_IMAGE_PATH_, imgId)
        pickle_path = join(_TRAIN_PICKLE_PATH_, pickle_name)
    else:
        image_path = join(_VAL_IMAGE_PATH_, imgId)
        pickle_path = join(_VAL_PICKLE_PATH_, pickle_name)

    #input(image_path)
    image = cv2.imread(image_path)
    image = image.astype(np.float32) / 255 - 0.5
    with open(pickle_path, 'rb') as heat_pickle:
        heatmap = pickle.load(heat_pickle)
    return image, heatmap


def _set_shapes(img, heatmap):
    img.set_shape([CONFIG['input_height'], CONFIG['input_width'], 3])
    heatmap.set_shape(
        [CONFIG['input_height'] / CONFIG['scale'], CONFIG['input_width'] / CONFIG['scale'], CONFIG['n_kpoints']])
    return img, heatmap


def _get_dataset_pipeline(anno, batch_size, epoch, buffer_size, is_train=True):

    images = os.listdir(_TRAIN_IMAGE_PATH_)
    dataset = tf.data.Dataset.from_tensor_slices(images)
    #imgIds = anno.getImgIds()

    #dataset = tf.data.Dataset.from_tensor_slices(imgIds)

    dataset.shuffle(20)
    dataset = dataset.map(
        lambda imgId: tuple(
            tf.py_func(
                func=_parse_function,
                inp=[imgId, is_train],
                Tout=[tf.float32, tf.float32]
            )
        ), num_parallel_calls=CONFIG['multiprocessing_num'])

    dataset = dataset.map(_set_shapes, num_parallel_calls=CONFIG['multiprocessing_num'])
    dataset = dataset.batch(64).repeat()
    dataset = dataset.prefetch(16)

    return dataset


def get_train_dataset_pipeline(batch_size=8, epoch=10, buffer_size=1):
    #global TRAIN_ANNO
    print('IN GET_TRAIN_PIPELINE')
    '''
    anno_path = join(BASE_PATH, TRAIN_JSON)
    print("preparing annotation from:", anno_path)
    TRAIN_ANNO = COCO(
        anno_path
    )
    '''
    return _get_dataset_pipeline(batch_size, epoch, buffer_size, True)

'''
def get_valid_dataset_pipeline(batch_size=8, epoch=10, buffer_size=1):
    print('IN GET_VAL_PIPELINE')
    global VALID_ANNO

    anno_path = join(BASE_PATH, VALID_JSON)
    print("preparing annotation from:", anno_path)
    VALID_ANNO = COCO(
        anno_path
    )
    return _get_dataset_pipeline(VALID_ANNO, batch_size, epoch, buffer_size, False)
'''