# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import cifar10
from datasets import flowers
from datasets import imagenet
from datasets import mnist
from datasets import visualwakewords

# NEED TO FIXED THESE DATASET LOADING
datasets_map = {
    'flowers': flowers,
    'mnist': mnist,
    'visualwakewords': visualwakewords,
}

datasets_map_modified = {
    'cifar10': cifar10,
    'imagenet': imagenet,
}

def get_dataset(name, split_name, dataset_dir, file_pattern=None, cycle_length=2):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A tf.data.Dataset

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map_modified:
    raise ValueError('Name of dataset unknown %s' % name)
  
  return datasets_map_modified[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      cycle_length)
