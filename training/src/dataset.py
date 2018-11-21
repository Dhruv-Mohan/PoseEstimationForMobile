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
_WIDTH_ = 128
_POINTS_ = 48
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
import random
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
    iaa.Affine(scale=(0.90, 1.11), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, rotate=(-5, 5), shear=(-1, 1), mode=['edge']),
    iaa.Fliplr(0.5)
    #iaa.pad()
    # iaa.CoarseDropout(0.2, size_percent=(0.001, 0.2))
], random_order=True)



BASE = "/root/hdd"
BASE_PATH = ""
#TRAIN_JSON = "train_gm16k.json"stea
#VALID_JSON = "val_gm16k.json"

_SBR_ = False
_EDGE_ = False
TRAIN_ANNO = None
VALID_ANNO = None
CONFIG = None
#_TRAIN_IMAGE_PATH_ = '/media/dhruv/Blue1/Blue1/Datasets/Menpo512/mobile_train_images_256/'
#_TRAIN_PICKLE_PATH_ = '/media/dhruv/Blue1/Blue1/Datasets/Menpo512/mobile_train_256_32/'

_TRAIN_IMAGE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_27k/Source/Menpo512_25/mobile_train_images_256/'
_TRAIN_PICKLE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_27k/Source/Menpo512_25/mobile_train_256_32/'
_TRAIN_IMAGE_PATH_  = '/home/dhruv/Projects/Datasets/Groomyfy_27k/Source/Menpo512_25/mobile_train_images_256_single/'
_SBR_PATH_ = '/media/dhruv/Blue1/Blue1/300VW_Dataset_2015_12_14/'
#_VAL_IMAGE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/mobile_val_images_256/'
#_VAL_PICKLE_PATH_ = '/home/dhruv/Projects/Datasets/Groomyfy_16k/Menpo51220/mobile_val_256/'
def set_config(config):
    global CONFIG, BASE, BASE_PATH
    CONFIG = config
    BASE = CONFIG['imgpath']
    BASE_PATH = CONFIG['datapath']

def _get_sobel_map(image):
    greyim = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(greyim, cv2.CV_32F)
    laplacian_min = np.min(laplacian)
    laplacian = laplacian - laplacian_min
    laplacian_max = np.max(laplacian)
    laplacian  = laplacian / laplacian_max + 1
    laplacian  = cv2.resize(laplacian, (32,32), 0)
    laplacian = np.expand_dims(laplacian, -1)
    laplacian = np.tile(laplacian, 91)

    return laplacian


def _parse_function(imgId, is_train, ann=None):
    imgId = imgId.decode()

    pickle_name = os.path.splitext(imgId)[0] + '.pickle'
    if _SBR_:
        pickle_name =  _get_sbr_pickle(imgId)
    if is_train == True:
        image_path = join(_TRAIN_IMAGE_PATH_, imgId)
        pickle_path = join(_TRAIN_PICKLE_PATH_, pickle_name)
    else:
        image_path = join(_VAL_IMAGE_PATH_, imgId)
        pickle_path = join(_VAL_PICKLE_PATH_, pickle_name)

    #input(image_path)
    image = cv2.imread(image_path)
    if _EDGE_:
        lap_edge = _get_sobel_map(image)
    image = image.astype(np.float32)
    image = cv2.resize(image, (_WIDTH_, _WIDTH_), 0)
    with open(pickle_path, 'rb') as heat_pickle:
        heatmap = pickle.load(heat_pickle)

    if _SBR_:
        heatmap_s1 = np.transpose(heatmap[0], [1,2,0])
        heatmap_s2 = np.transpose(heatmap[1], [1,2,0])
        heatmap_s3 = np.transpose(heatmap[2], [1,2,0])
    else:
        heatmap_s1 = heatmap
        heatmap_s2 = heatmap
        heatmap_s3 = heatmap

    if _EDGE_:
        heatmap_s1 = np.multiply(heatmap_s1, lap_edge)
        heatmap_s2 = np.multiply(heatmap_s2, lap_edge)
        heatmap_s3 = np.multiply(heatmap_s3, lap_edge)

    heatmap_s1 = heatmap_s1[:,:, 0:_POINTS_]
    heatmap_s2 = heatmap_s2[:, :, 0:_POINTS_]
    heatmap_s3 = heatmap_s3[:, :, 0:_POINTS_]
    aug_det = aug.to_deterministic()
    image = aug_det.augment_image(image)
    aug_heatmaps1 = aug_det.augment_image(heatmap_s1)
    aug_heatmaps2 = aug_det.augment_image(heatmap_s2)
    aug_heatmaps3 = aug_det.augment_image(heatmap_s3)
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
    return image, aug_heatmaps1, aug_heatmaps2, aug_heatmaps3


def _set_shapes(img, heatmap1, heatmap2, heatmap3):
    img.set_shape([CONFIG['input_height'], CONFIG['input_width'], 3])
    heatmap1.set_shape(
        [32, 32, _POINTS_])
    heatmap2.set_shape(
        [32, 32, _POINTS_])
    heatmap3.set_shape(
        [32, 32, _POINTS_])
    return img, heatmap1, heatmap2, heatmap3

def _get_sbr_images():
    folders = os.listdir(_SBR_PATH_)
    images = []
    for folder in folders:
        folder_path = os.path.join(_SBR_PATH_, folder)
        image_path = os.path.join(folder_path, 'imgs')
        img_names = os.listdir(image_path)
        for image_name in img_names:
            full_image_path = os.path.join(image_path, image_name)
            images.append(full_image_path)

    random.shuffle(images)
    return images

def _get_sbr_pickle(Image_path):
    image_folder, image_name = os.path.split(Image_path)
    pickle_name = os.path.splitext(image_name)[0] + '.pickle'
    pickle_folder = os.path.split(image_folder)[0] + '/pickles'
    pickle_path = os.path.join(pickle_folder, pickle_name)
    return pickle_path

def _get_dataset_pipeline(anno, batch_size, epoch, buffer_size, is_train=True):

    images = os.listdir(_TRAIN_IMAGE_PATH_)
    if _SBR_:
        images = _get_sbr_images()
    dataset = tf.data.Dataset.from_tensor_slices(images)

    #imgIds = anno.getImgIds()

    #dataset = tf.data.Dataset.from_tensor_slices(imgIds)

    dataset.shuffle(20)
    dataset = dataset.map(
        lambda imgId: tuple(
            tf.py_func(
                func=_parse_function,
                inp=[imgId, is_train],
                Tout=[tf.float32, tf.float32,tf.float32,tf.float32]
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