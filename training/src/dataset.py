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
import imgaug as ia
from imgaug import augmenters as iaa

aug = iaa.SomeOf((0, None), [
    # iaa.AdditiveGaussianNoise(scale=(0, 0.002)),
    iaa.Noop(),
iaa.Noop(),
iaa.Noop(),
iaa.Noop(),
    #iaa.GaussianBlur(sigma=(0.0, 0.15)),
    #iaa.Dropout(p=(0, 0.02)),
    #iaa.AddElementwise((-10, 10), per_channel=0.5),
    #iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 0.1)),
    # iaa.ContrastNormalization((0.5, 1.5)),
    iaa.Affine(scale=(0.90, 1.11), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, rotate=(-5, 5), shear=(-1, 1), mode=['edge'])
    #iaa.pad()
    # iaa.CoarseDropout(0.2, size_percent=(0.001, 0.2))
], random_order=True)



BASE = "/root/hdd"
BASE_PATH = ""
#TRAIN_JSON = "train_gm16k.json"stea
#VALID_JSON = "val_gm16k.json"

TRAIN_ANNO = None
VALID_ANNO = None
CONFIG = None
_TRAIN_IMAGE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_27k/Source/Menpo512_25/mobile_train_images_256/'
_TRAIN_PICKLE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_27k/Source/Menpo512_25/mobile_train_256_32/'

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
    image = image.astype(np.float32)
    with open(pickle_path, 'rb') as heat_pickle:
        heatmap = pickle.load(heat_pickle)

    aug_det = aug.to_deterministic()
    image = aug_det.augment_image(image)
    aug_heatmaps2 = aug_det.augment_image(heatmap)
    '''
    aug_heatmaps = []
    for i in range(91):
        single_heatmap = heatmap[:, :, i]
        single_heatmap = np.expand_dims(single_heatmap, -1)
        aug_heatmap = aug_det.augment_image(single_heatmap)
        #aug_heatmap = np.squeeze(aug_heatmap)
        aug_heatmaps.append(aug_heatmap)
        cv2.imshow('aug_map', cv2.resize(aug_heatmap, (512,512), 0))
        #cv2.imshow('single_map', cv2.resize(single_heatmap, (512, 512),0))
        #cv2.waitKey(0)

    aug_heatmaps = np.concatenate(aug_heatmaps, axis=-1)
    '''
    #input(heatmap.shape)
    #input(aug_heatmaps.shape)
    return image, aug_heatmaps2


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
    dataset = dataset.batch(32).repeat()
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