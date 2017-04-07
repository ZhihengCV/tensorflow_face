"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from nets import nets_factory
import time
import lfw
import utils
from datetime import datetime

tf.app.flags.DEFINE_string(
    'model_name', 'vgg_flip', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string('lfw_dir', '/media/teddy/data/lfw_array_112',
                           """Where to save lfw image"""
                           )

tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 500,
                            """How often to run the eval.""")


tf.app.flags.DEFINE_string(
    'checkpoint_dir', '../vgg_flip_sparse',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '../vgg_flip_sia_eval', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_string(
    'eval_mode', 'online', 'offline/online/once')

tf.app.flags.DEFINE_string(
    'device', '/gpu:0', 'cpu:0 or gpu:0')

FLAGS = tf.app.flags.FLAGS


def _cal_acc(embedding, label_arr):
    assert len(embedding) == len(label_arr) == 12000, "wrong feature num"
    fea_l = embedding[:6000]
    fea_r = embedding[6000:]
    assert label_arr[:6000].all() == label_arr[6000:].all(), "wrong label order"
    labels = label_arr[:6000]
    scores = utils.cos(fea_l, fea_r)
    return utils.best_acc(scores, labels)


def eval_once(global_step, lfw_data, sess, images_placeholder,
              embedding, summary_op, summary_writer):
    """Run Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """

    print("eval model at {} steps".format(global_step))
    feature_list = []
    label_list = []

    assert 12000 % FLAGS.batch_size == 0, "12000 should be divided by batch_size"
    val_iter_num = 12000 // FLAGS.batch_size

    for num in xrange(int(val_iter_num)):
        if num < int(val_iter_num) - 1:
            tensor_list = [embedding]
        else:
            tensor_list = [embedding, summary_op]

        images, labels = lfw_data.next_batch(FLAGS.batch_size)
        out_list = sess.run(tensor_list,
                            feed_dict={images_placeholder: images})
        label_list.append(labels)
        feature_list.append(out_list[0])

    feas = np.vstack(feature_list)
    labs = np.concatenate(label_list)
    acc, thre = _cal_acc(feas, labs)
    summary = tf.Summary()
    summary.ParseFromString(out_list[1])
    print('%s: best acc = %.3f @ %.3f' % (datetime.now(), acc, thre))
    summary.value.add(tag='lfw_acc', simple_value=acc)
    summary_writer.add_summary(summary, global_step)

    return acc

def weight_summary(weights):
    for weight in weights:
        name = weight.op.name
        mean = tf.reduce_mean(weight)
        tf.summary.scalar('%s/mean' % name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(weight - mean)))
        tf.summary.scalar('%s/stddev' % name, stddev)
        tf.summary.histogram(name + '/distribute', weight)


def activation_summary(end_points):
    for end_point in end_points:
        x = end_points[end_point]
        tf.summary.histogram(x.op.name + '/activations', x)
        tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        ####################
        # set up input#
        ####################
        lfw_data = lfw.LFWData(FLAGS.lfw_dir)
        ####################
        # Select the network #
        ####################
        network_fn_val = nets_factory.get_network_val(FLAGS.model_name)
        ####################
        # Define the model #
        ####################
        with tf.device(FLAGS.device):
            val_shape = (FLAGS.batch_size, lfw.HEIGHT, lfw.WIDTH, 1)
            images_placeholder = tf.placeholder(tf.float32, shape=val_shape, name='val_images')
            embedding, end_points = network_fn_val(images_placeholder)
        activation_summary(end_points)
        weights = tf.trainable_variables()
        weight_summary(weights)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

        if FLAGS.eval_mode not in ['offline', 'online', 'once']:
            raise ValueError("mode should be one of offline/online/once")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if FLAGS.eval_mode == 'offline':
                model_path_list = ckpt.all_model_checkpoint_paths
                for model_path in model_path_list:
                    global_step = model_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, model_path)
                    eval_once(global_step, lfw_data, sess, images_placeholder,
                              embedding, summary_op, summary_writer)
            elif FLAGS.eval_mode == 'once':
                model_path = ckpt.model_checkpoint_path
                global_step = model_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, model_path)
                eval_once(global_step, lfw_data, sess, images_placeholder,
                          embedding, summary_op, summary_writer)
            else:
                old_model_list = []
                while True:
                    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                    temp_model_list = ckpt.all_model_checkpoint_paths
                    new_item = [i for i in temp_model_list if i not in old_model_list]
                    for model_path in new_item:
                        global_step = model_path.split('/')[-1].split('-')[-1]
                        saver.restore(sess, model_path)
                        eval_once(global_step, lfw_data, sess, images_placeholder, embedding, summary_op, summary_writer)
                    old_model_list = temp_model_list
                    time.sleep(FLAGS.eval_interval_secs)
            sess.close()
        else:
            print('No checkpoint file found')
            return

if __name__ == '__main__':
    tf.app.run()
