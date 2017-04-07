import os
import cv2
import tensorflow as tf
import random


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
       a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
    values: A scalar or list of values.

    Returns:
       a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def image_to_tfexample(image_data, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/class/label': int64_feature(class_id),
    }))


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/webface_%s.tfrecord' % (dataset_dir, split_name)


def _batch_write(input_list, tfrecord_writer):
    random.shuffle(input_list)
    for i, l in input_list:
        example = image_to_tfexample(i, l)
        tfrecord_writer.write(example.SerializeToString())

def _image_in_bytes(iamge_path):
    with open(iamge_path, "rb") as f:
        byte_f = f.read()
    return byte_f

def _add_to_tfrecord(image_dir, train_writer, val_writer, max_cache = 100000):

    train_im_label_list = []
    val_im_label_list = []
    id_count = 0
    train_count = 0
    val_count = 0
    cache_count = 0
    for parent, _, filenames in os.walk(image_dir, topdown=False):
        img_filenames = [i for i in filenames if i.find('.jpg')]
        if len(img_filenames) > 0:
            label = int(parent.rsplit('/', 1)[-1].strip())
            print('process {} id'.format(label))
            id_count += 1
            for img_filename in img_filenames[:-1]:
                img_path = os.path.join(parent, img_filename)
                train_im_label_list.append((_image_in_bytes(img_path), label))
                train_count += 1
                cache_count += 1
            img_path = os.path.join(parent, img_filenames[-1])
            val_im_label_list.append((_image_in_bytes(img_path), label))
            val_count += 1
        if cache_count > max_cache:
            _batch_write(train_im_label_list, train_writer)
            train_im_label_list = []
            cache_count = 0
    _batch_write(val_im_label_list, val_writer)
    if len(train_im_label_list) > 0:
        _batch_write(train_im_label_list, train_writer)

    print "finish all"
    return id_count, train_count, val_count


def run(dataset_dir, image_dir):
    """Runs the download and conversion operation.

    Args:
       dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train')
    val_filename = _get_output_filename(dataset_dir, 'val')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(val_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    train_writer = tf.python_io.TFRecordWriter(training_filename)
    val_writer = tf.python_io.TFRecordWriter(val_filename)
    id_count, train_count, val_count = _add_to_tfrecord(image_dir, train_writer, val_writer)
    train_writer.close()
    val_writer.close()
    with open(os.path.join(dataset_dir, 'casia_clean_disc.txt'), 'w') as f:
        f.write("casia clean version, rotate only crop method, size 96*112*3\n")
        f.write("There are {} ids in the dataset\n".format(id_count))
        f.write("There are {} images in the train\n".format(train_count))
        f.write("There are {} images in the val\n".format(val_count))
    print('finish!')

if __name__ == '__main__':
    dataset_dir = '/media/teddy/data/casia_clean_tf'
    image_dir = '/media/teddy/data/casia_crop'
    run(dataset_dir, image_dir)