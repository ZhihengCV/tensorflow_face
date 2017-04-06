# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Read and preprocess image data.

 This script is designed to process Face Data which is already aligned\

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_height', 112,
                            """Provide the height of images""")
tf.app.flags.DEFINE_integer('image_width', 96,
                            """Provide the  of images""")
tf.app.flags.DEFINE_integer('image_channel', 3,
                            """Provide the channel of images""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads"""
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=FLAGS.image_channel)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

       The output of the build_image_data.py image preprocessing script is a dataset
       containing serialized Example protocol buffers.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
    }

    with tf.name_scope('decode_tfrecord'):
        features = tf.parse_single_example(example_serialized, feature_map)
        image = decode_jpeg(features['image/encoded'])
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        return image, label


def image_preprocessing(image, train):
    """Decode and preprocess one image for evaluation or training.

    Args:
      image: JPEG
      train: boolean
    Returns:
      3-D float Tensor containing an appropriately scaled image

    Raises:
       ValueError: if user does not provide bounding box
    """
    with tf.name_scope('image_preprocessing'):
        if train:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.6)
            if FLAGS.image_channel >= 3:
                image = tf.image.random_saturation(image, 0.6, 1.4)
        # Finally, rescale to [-1,1] instead of [0, 1)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        image = tf.image.per_image_standardization(image)
        return image


def train_inputs(dataset, batch_size=None, num_preprocess_threads=None):
    """Generate batches of distorted versions of ImageNet images.

        Use this function as the inputs for training a network.

        Distorting images provides a useful technique for augmenting the data
        set during training in order to make the network invariant to aspects
        of the image that do not effect the label.

        Args:
        dataset: instance of Dataset class specifying the dataset.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but
          None defaults to FLAGS.num_preprocess_threads.

        Returns:
        images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                           FLAGS.image_size, 1].
        labels: 1-D integer Tensor of [batch_size].
    """
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    if not batch_size:
        batch_size = FLAGS.batch_size
    with tf.device('/cpu:0'):
        with tf.variable_scope("train_input"):
            images, labels = batch_inputs(
                dataset, batch_size, train=True,
                num_preprocess_threads=num_preprocess_threads,
                num_readers=FLAGS.num_readers)
    return images, labels


def eval_inputs(dataset, batch_size, num_preprocess_threads=None):
    """Generate batches of distorted versions of ImageNet images.

        Use this function as the inputs for training a network.

        Distorting images provides a useful technique for augmenting the data
        set during training in order to make the network invariant to aspects
        of the image that do not effect the label.

        Args:
        dataset: instance of Dataset class specifying the dataset.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but
          None defaults to FLAGS.num_preprocess_threads.

        Returns:
        images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                           FLAGS.image_size, 1].
        labels: 1-D integer Tensor of [batch_size].
    """
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        if not batch_size:
            batch_size = FLAGS.batch_size
        with tf.variable_scope("val_input"):
            images, labels = batch_inputs(
                dataset, batch_size, train=False,
                num_preprocess_threads=num_preprocess_threads,
                num_readers=FLAGS.num_readers)
    return images, labels


def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None,
                 num_readers=None):
    """Contruct batches of training or evaluation examples from the image dataset.

    Args:
    dataset: instance of Dataset class specifying the dataset.
      See dataset.py for details.
    batch_size: integer
    train: boolean
    num_preprocess_threads: integer, total number of preprocessing threads
    num_readers: integer, number of parallel readers

    Returns:
      images: 4-D float Tensor of a batch of images
      labels: 1-D integer Tensor of [batch_size].

    Raises:
      ValueError: if data is not found
    """
    with tf.name_scope('batch_processing'):
        data_files = dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue, and decide shuffle or not
        if train:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=True,
                                                            capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=False,
                                                            capacity=1)

        if num_preprocess_threads is None:
            num_preprocess_threads = FLAGS.num_preprocess_threads

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers is None:
            num_readers = FLAGS.num_readers

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 112*96*1*4 bytes = 0.08MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 0.08MB = 1.4GB
        min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
        if train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 3 * batch_size,
                dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        with tf.name_scope("example_reader"):
            if num_readers > 1:
                enqueue_ops = []
                for _ in range(num_readers):
                    reader = dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.add_queue_runner(tf.train.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                reader = dataset.reader()
                _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image, label = parse_example_proto(example_serialized)
            image = image_preprocessing(image, train)
            image.set_shape([dataset.height, dataset.width, dataset.depth])
            images_and_labels.append([image, label])

        images, label_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size)

        # Reshape images into these desired dimensions.
        height = FLAGS.image_height
        width = FLAGS.image_width
        depth = FLAGS.image_channel

        # check the input shape and whether it is converted to gray
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        return images, tf.reshape(label_batch, [batch_size])
