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
import argparse
from networks import get_network
import os

_DSP_ = True

from pprint import pprint

os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--model', type=str, default='mv2_cpm', help='')
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--checkpoint', type=str, default='/home/dhruv/Projects/PersonalGit/PoseEstimationForMobile/training/model/crap/model-5002.index', help='checkpoint path')
parser.add_argument('--output_node_names', type=str, default='Convolutional_Pose_Machine/output')
parser.add_argument('--output_graph', type=str, default='./v3.1_trained_augment.pb', help='output_freeze_path')

args = parser.parse_args()

input_node = tf.placeholder(tf.float32, shape=[1, args.size, args.size, 3], name="image")


with tf.Session() as sess:
    net = get_network(args.model, input_node,  trainable=False)
    saver = tf.train.Saver()
    latest_ckpt = tf.train.latest_checkpoint('model/mv2_cpm_batch-64_lr-0.0005_gpus-1_256x256_experiments-mv2_cpm/')
    saver.restore(sess, latest_ckpt)
    print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])

    from tensorflow.tools.graph_transforms import TransformGraph

    transforms = ['add_default_attributes',
                  'strip_unused_nodes(type=float, shape="1,200,200,3")',
                  'strip_unused_nodes(type=float, shape="1,200,200,3")',
                  'fold_constants(ignore_errors=true)',
                  'remove_nodes(op=Identity, op=CheckNumerics)',
                  'fold_batch_norms', 'fold_old_batch_norms',
                  'remove_control_dependencies',
                  'strip_unused_nodes', 'sort_by_execution_order']
    '''
    transforms = ['add_default_attributes',
                  'strip_unused_nodes(type=float, shape="1,256,256,3")',
                  'strip_unused_nodes(type=float, shape="1,256,256,3")',
                        'remove_nodes(op=Identity, op=CheckNumerics, op=Rsqrt)',
                        'fold_constants(ignore_errors=true)',
                        'fold_batch_norms',
                        'fold_old_batch_norms',
                        'backport_concatv2',
                  'quantize_weights(minimum_size=2)',
                  'quantize_nodes',
                        'remove_control_dependencies',
                        'strip_unused_nodes',
                        'sort_by_execution_order']
                        
     '''
    transformed_graph_def = TransformGraph(tf.get_default_graph().as_graph_def(), 'Placeholder',
                                           args.output_node_names.split(","), transforms)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        transformed_graph_def,  # The graph_def is used to retrieve the nodes
        args.output_node_names.split(",")  # The output node names are used to select the useful nodes
    )
    _NAME_ = "3.6.28.pb"
    with tf.gfile.GFile(_NAME_, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    import shutil
    input_path = '/home/dhruv/Projects/PersonalGit/PoseEstimationForMobile/training/' + _NAME_
    output_path = '/home/dhruv/Projects/PersonalGit/PoseEstimationForMobile/android_demo/demo_mace/' + _NAME_
    shutil.copy(input_path, output_path)

    print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])
    input_graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        args.output_node_names.split(",")
    )
    
    

with tf.gfile.GFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
