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
"""Provides utilities to Cifar-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.vision.image_classification import imagenet_preprocessing

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1

# TODO(tobyboyd): Change to best practice 45K(train)/5K(val)/10K(test) splits.
NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}
_NUM_DATA_FILES = 5
NUM_CLASSES = 10


def get_parse_record_fn(use_keras_image_data_format=False):
  """Get a function for parsing the records, accounting for image format.

  This is useful by handling different types of Keras models. For instance,
  the current resnet_model.resnet50 input format is always channel-last,
  whereas the keras_applications mobilenet input format depends on
  tf.keras.backend.image_data_format(). We should set
  use_keras_image_data_format=False for the former and True for the latter.

  Args:
    use_keras_image_data_format: A boolean denoting whether data format is keras
      backend image data format. If False, the image format is channel-last. If
      True, the image format matches tf.keras.backend.image_data_format().

  Returns:
    Function to use for parsing the records.
  """
  def parse_record_fn(raw_record, is_training, dtype):
    image, label = parse_record(raw_record, is_training, dtype)
    if use_keras_image_data_format:
      if tf.keras.backend.image_data_format() == 'channels_first':
        image = tf.transpose(image, perm=[2, 0, 1])
    return image, label
  return parse_record_fn

def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  This method converts the label to one hot to fit the loss function.

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: Data type to use for input images.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image = raw_record['image']
  label = raw_record['label']
  label = tf.cast(label, tf.int32)

  image = preprocess_image(image, is_training)
  image = tf.cast(image, dtype)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_with_crop_or_pad(
        image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  assert tf.io.gfile.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def input_fn(is_training,
             data_dir,
             batch_size,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record,
             input_context=None,
             drop_remainder=False,
             training_dataset_cache=False,
             tf_data_experimental_slack=False,
             filnames=None):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    parse_record_fn: Function to use for parsing the records.
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

  Returns:
    A dataset that can be used for iteration.
  """
  cifar_builder = tfds.builder('cifar10', data_dir=data_dir)
  if len(os.listdir(data_dir)) == 0:
    cifar_builder.download_and_prepare()

  if is_training:
    ds = cifar_builder.as_dataset(split='train')
  else:
    ds = cifar_builder.as_dataset(split='test')

  return imagenet_preprocessing.process_record_dataset(
      dataset=ds,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=NUM_IMAGES['train'],
      parse_record_fn=parse_record_fn,
      dtype=dtype,
      datasets_num_private_threads=datasets_num_private_threads,
      drop_remainder=drop_remainder,
      tf_data_experimental_slack=tf_data_experimental_slack
  )
