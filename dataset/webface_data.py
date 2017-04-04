
"""Small library that points to the ImageNet data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.dataset_base import Dataset
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

# Where to load the data
tf.app.flags.DEFINE_string('data_dir', '/media/teddy/data/casia_clean_tf',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")


class WebfaceData(Dataset):
    """ImageNet data set."""

    def __init__(self, subset):
        super(WebfaceData, self).__init__('Webface', subset)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 10575

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        # Bounding box data consists of 615299 bounding boxes for 544546 images.
        if self.subset == 'train':
            return 445019
        else:
            return 10575

    def data_files(self):
        """Returns a python list of all (sharded) data subset files.

          Returns:
            python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        tf_record_pattern = os.path.join(FLAGS.data_dir,
                                         "webface_{}*".format(self.subset))
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print("No file is found ")
        return data_files

    @property
    def height(self):
        return 112

    @property
    def width(self):
        return 96

    @property
    def depth(self):
        return 3
