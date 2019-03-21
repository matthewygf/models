# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Cifar10 dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_cifar10.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

_FILE_PATTERN = 'cifar10_%s.tfrecord'

SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}

_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 9',
}

def parse_fn(data):
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  parsed = tf.parse_single_example(data, keys_to_features)
  image = tf.io.decode_image(parsed["image/encoded"], channels=3)
  label = parsed["image/class/label"] 
  # haven't figured out how generalize the data map, becuase we are aiming generalize lol
  return image, label, () 

def get_split(split_name, dataset_dir, file_pattern=None, cycle_length=2):
  """Gets a dataset tuple with instructions for reading cifar10.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    dataset - of type tf.data.Dataset

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
  print("file path: %s" % file_pattern) 
  assert tf.gfile.Exists(file_pattern)
  print("file exists -- continue getting data")

  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)
  print(labels_to_names)

  # declare dataset
  files = tf.data.Dataset.list_files([file_pattern])
  train_dataset = files.apply(tf.data.experimental.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=cycle_length
    )
  )
  train_dataset = train_dataset.map(map_func=parse_fn, num_parallel_calls=cycle_length)
  return train_dataset, _NUM_CLASSES, SPLITS_TO_SIZES[split_name]

