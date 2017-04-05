"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import model_map
from dataset import data_map
from input_pipline import eval_inputs
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import math

tf.app.flags.DEFINE_string(
    'model_name', 'squeezenet', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'data_name', 'webface', 'The name of the data')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0007,
                          'The weight decay on the model weights.')
tf.app.flags.DEFINE_float('label_smoothing', 0.0,
                          """The amount of label smoothing.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 300,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_string(
    'checkpoint_dir', './train_result/squeezenet_new',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

# tf.app.flags.DEFINE_string(
#     'eval_dir', './single_gpu', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_string(
    'eval_mode', 'online', 'offline/online/once')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.3,
                          """Upper bound on the amount of GPU memory
                          that will be used by the process""")

tf.app.flags.DEFINE_string(
    'device', '/gpu:0', 'cpu:0 or gpu:0')


FLAGS = tf.app.flags.FLAGS


def eval_once(global_step, sess, val_loss, top_one_op, top_five_op, image_num):
    """Run Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    print("eval model as {} steps".format(global_step))

    true_one_count = 0  # Counts the number of correct predictions.
    true_five_count = 0
    loss_sum = 0
    iter_num = math.ceil(float(image_num) / float(FLAGS.batch_size))
    total_sample_count = int(iter_num * FLAGS.batch_size)

    step = 0
    while step < iter_num:
        loss_v, pre_one, pre_five = sess.run([val_loss, top_one_op, top_five_op])
        true_one_count += np.sum(pre_one)
        true_five_count += np.sum(pre_five)
        loss_sum += loss_v
        step += 1

    # Compute precision @ 1 @ 5.
    loss_mean = loss_sum / iter_num
    print('%s: val_loss = %.3f' % (datetime.now(), loss_mean))
    precision = 1.0 * true_one_count / total_sample_count
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    precision = 1.0 * true_five_count / total_sample_count
    print('%s: precision @ 5 = %.3f' % (datetime.now(), precision))


def main(_):
    if FLAGS.eval_mode not in ['offline', 'online', 'once']:
        raise ValueError("mode should be one of offline/online/once")

    with tf.Graph().as_default():
        ####################
        # set up input#
        ####################
        model = model_map[FLAGS.model_name]
        val_dataset = data_map[FLAGS.data_name]('val')
        val_images, val_labels = eval_inputs(val_dataset, FLAGS.batch_size)
        num_classes = val_dataset.num_classes()
        image_num = val_dataset.num_examples_per_epoch()

        ####################
        # Define the model #
        ####################
        with tf.device(FLAGS.device):
            val_logits = model.inference(val_images, num_classes, is_training=False)
            val_loss = model.loss(val_logits, val_labels)
            top_one_op = tf.nn.in_top_k(val_logits, val_labels, 1)
            top_five_op = tf.nn.in_top_k(val_logits, val_labels, 5)

        saver = tf.train.Saver(tf.global_variables())
        # Build the summary operation based on the TF collection of Summaries.

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = FLAGS.log_device_placement
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
        sess = tf.Session(config=config)

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            if FLAGS.eval_mode == 'offline':
                model_path_list = ckpt.all_model_checkpoint_paths
                for model_path in model_path_list:
                    global_step = model_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, model_path)
                    eval_once(global_step, sess, val_loss, top_one_op, top_five_op, image_num)
            elif FLAGS.eval_mode == 'once':
                model_path = ckpt.model_checkpoint_path
                global_step = model_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, model_path)
                eval_once(global_step, sess, val_loss, top_one_op, top_five_op, image_num)
            else:
                old_model_list = []
                while True:
                    time.sleep(FLAGS.eval_interval_secs)
                    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                    temp_model_list = ckpt.all_model_checkpoint_paths
                    new_item = [i for i in temp_model_list if i not in old_model_list]
                    for model_path in new_item:
                        global_step = model_path.split('/')[-1].split('-')[-1]
                        saver.restore(sess, model_path)
                        eval_once(global_step, sess, val_loss, top_one_op, top_five_op)
                    old_model_list = temp_model_list
            coord.request_stop()
            coord.join(threads)
            sess.close()
        else:
            print('No checkpoint file found')
            return

if __name__ == '__main__':
    tf.app.run()
