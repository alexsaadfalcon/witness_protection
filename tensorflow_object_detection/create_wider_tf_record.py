# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import glob

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/derin/datasets/WIDER',
                    'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train',
                    'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('output_path', '/home/derin/datasets/WIDER/tfrecord',
                    'Path to output TFRecord')
flags.DEFINE_string('label_map_path',
                    '/home/derin/datasets/WIDER/wider_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_int('shards', '5', 'Number of shards for dataset.')
flags.DEFINE_string('dataset', 'WIDER', 'Name of dataset.')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       split):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join('WIDER_' + split, 'images', data['folder'],
                          data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  if 'object' in data:
    for obj in data['object']:


      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))

  data_dir = FLAGS.data_dir
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  examples_path = os.path.join(data_dir, 'WIDER_' + FLAGS.set, 'images')
  examples_subdirs = [dI for dI in os.listdir(examples_path) if
                   os.path.isdir(os.path.join(examples_path, dI))]
  examples_list = []
  for subdir in examples_subdirs:
      examples_list.extend(glob.glob(os.path.join(examples_path,
                                                  subdir, '*.jpg')))


  for i in range(len(examples_list)):
      examples_list[i] = examples_list[i].split(os.sep)[-1]
      examples_list[i] = examples_list[i].split('.jpg')[0]

  annotations_dir = os.path.join(data_dir, 'WIDER_' + FLAGS.set, 'annotations')
  
  num_images = len(examples_list)
  num_per_shard = int(math.ceil(num_images / float(FLAGS.shards)))

  for shard_id in range(FLAGS.shards):
    output_filename = os.path.join(FLAGS.output_path, '%s-%05d-of-%05d.tfrecord' % (FLAGS.dataset, shard_id, FLAGS.shards))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for idx in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        path = os.path.join(annotations_dir, example + '.xml')
        with tf.gfile.GFile(path, 'r') as fid:
          xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                    FLAGS.set)
        tfrecord_writer.write(tf_example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()
  
  #for idx, example in enumerate(examples_list):
  #  if idx % 100 == 0:
  #    logging.info('On image %d of %d', idx, len(examples_list))
  #  path = os.path.join(annotations_dir, example + '.xml')
  #  with tf.gfile.GFile(path, 'r') as fid:
  #    xml_str = fid.read()
  #  xml = etree.fromstring(xml_str)
  #  data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

  #  tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
  #                                  FLAGS.set)
  #  writer.write(tf_example.SerializeToString())

  #writer.close()


if __name__ == '__main__':
  tf.app.run()
