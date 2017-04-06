"""
script to train model using a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from datetime import datetime

from models import model_map
from dataset import data_map
from input_pipline import train_inputs
import tensorflow as tf
import numpy as np


FLAGS = tf.app.flags.FLAGS

# basic config about train model and data
tf.app.flags.DEFINE_string(
    'model_name', 'densenet', 'The name of the model')
tf.app.flags.DEFINE_string(
    'data_name', 'webface', 'The name of the data')
tf.app.flags.DEFINE_string('train_dir', './train_result/densenet_121',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a train batch.""")
tf.app.flags.DEFINE_integer('max_epoch', 40,
                            """Number of epoch to run.""")

# regular method
tf.app.flags.DEFINE_float('weight_decay', 0.0007,
                          'The weight decay on the model weights.')
tf.app.flags.DEFINE_float('label_smoothing', 0.0,
                          """The amount of label smoothing.""")

# model save policy
tf.app.flags.DEFINE_integer('save_step', 2500,
                            """Number of step to save model.""")
tf.app.flags.DEFINE_integer('max_model_num', 40,
                            """Number of newest model to keep""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('gpu_memory_fraction', 1.0,
                          """Upper bound on the amount of GPU memory
                          that will be used by the process""")
tf.app.flags.DEFINE_string('device', '/gpu:0', 'set the device name')

#Flags to indicate whether resume training or finetune
tf.app.flags.DEFINE_boolean('resume', False,
                            """If set, training will load model from appoint path """)
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('exclude_scopes', None,
                           """Comma-separated list of scopes of variables to
                               exclude when restoring from a checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# Flags governing the optimization.
tf.app.flags.DEFINE_string('optimizer', 'momentum',
                           """The name of the optimizer,
                              one of "momentum", "rmsprop".""")
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential',
                           """Specifies how the learning rate is decayed.
                           One of "fixed", "exponential",or "polynomial" """)

# Flags config the optimization
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001,
                          """The minimal end learning rate """)
tf.app.flags.DEFINE_float('num_epochs_per_decay', 10.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """The momentum for the MomentumOptimizer.""")
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, """Momentum for RMSProp.""")

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, """Decay term for RMSProp.""")

tf.app.flags.DEFINE_float('rmsprop_eplision', 1.0, """Decay term for RMSProp.""")


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.initial_learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _init_weight(sess):
    if FLAGS.resume and FLAGS.fine_tune:
        raise Exception("There should be only one mode")

    if FLAGS.fine_tune or FLAGS.resume:
        exclusions = []
        if FLAGS.exclude_scopes:
            exclusions = [scope.strip()
                          for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
        if FLAGS.resume:
            step = FLAGS.checkpoint_path.split('/')[-1].split('-')[-1]
            print("resume training from step {}".format(step))
        else:
            print("finetune")
            step = 0
            exclusions.append('global_step')
            exclusions.append('logits')
            exclusions.append('aux_logits')
        variables_to_restore = []
        variable_to_init = []
        for var in tf.global_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
            else:
                variable_to_init.append(var)
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, FLAGS.checkpoint_path)
        init_op = tf.group(*[v.initializer for v in variable_to_init])
        sess.run([init_op])
        print("finish load model")
        return int(step)
    else:
        print("train model from scratch")
        init = tf.global_variables_initializer()
        sess.run(init)
        return 0


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)
        ####################
        # set up input#
        ####################
        model = model_map[FLAGS.model_name]
        train_dataset = data_map[FLAGS.data_name]('train')
        train_images, train_labels = train_inputs(train_dataset, FLAGS.batch_size)
        num_classes = train_dataset.num_classes()
        #############################
        # Specify the loss function #
        #############################
        # forward,and transfer label to onehot_labels for label smoothing
        with tf.device(FLAGS.device):
            train_logits = model.inference(train_images, num_classes, is_training=True)
            train_loss = model.loss(train_logits, train_labels)
            top_1_op = tf.nn.in_top_k(train_logits, train_labels, 1)
            top_5_op = tf.nn.in_top_k(train_logits, train_labels, 5)
            # Gather update_ops from the first clone. These contain, for example,
            # the updates for the batch_norm variables created by network_fn.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            #########################################
            # Configure the optimization procedure. #
            #########################################
            learning_rate = _configure_learning_rate(train_dataset.num_examples_per_epoch(),
                                                     global_step)
            tf.summary.scalar('learning_rate', learning_rate)
            optimizer = _configure_optimizer(learning_rate)
            grads = optimizer.compute_gradients(train_loss)
            grad_updates = optimizer.apply_gradients(grads,
                                                     global_step=global_step)
            update_ops.append(grad_updates)
            # group all the update option
            with tf.control_dependencies(update_ops):
                train_op = tf.no_op(name='train')

        # add summary to supervise trainable variable and the gradient
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_model_num)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        #############################
        # Define the init function #
        #############################
        # Build an initialization operation to run below.
        # Start running operations on the Graph.
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = FLAGS.log_device_placement
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
        sess = tf.Session(config=config)
        step = _init_weight(sess)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        num_per_epoch = train_dataset.num_examples_per_epoch()
        num_examples_per_step = FLAGS.batch_size

        epoch = step * num_examples_per_step // num_per_epoch
        while epoch < FLAGS.max_epoch:
            start_time = time.time()

            if step % 100 == 0 and step % 500 != 0:
                loss_value, lr, top_1, top_5, _ = sess.run([train_loss, learning_rate,
                                                            top_1_op, top_5_op, train_op])

                duration = time.time() - start_time
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                top1_acc = np.sum(top_1) / num_examples_per_step
                top5_acc = np.sum(top_5) / num_examples_per_step
                format_str = ('%s: step %d epoch %d, loss = %.2f ,top1 acc = %.2f , top5 acc = %.2f '
                              '(%.1f examples/sec; %.3f sec/batch at learning rate %.6f')
                print(format_str % (datetime.now(), step, epoch, loss_value, top1_acc, top5_acc,
                                    examples_per_sec, sec_per_batch, lr))
            elif step % 500 == 0:
                # summary option is time consuming
                loss_value, lr, summary_str, top_1, top_5, _ = sess.run([train_loss, learning_rate, summary_op,
                                                                         top_1_op, top_5_op, train_op])
                duration = time.time() - start_time
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                top1_acc = np.sum(top_1) / num_examples_per_step
                top5_acc = np.sum(top_5) / num_examples_per_step
                format_str = ('%s: step %d epoch %d, loss = %.2f ,top1 acc = %.2f , top5 acc = %.2f '
                              '(%.1f examples/sec; %.3f sec/batch at learning rate %.6f')
                print(format_str % (datetime.now(), step, epoch, loss_value, top1_acc, top5_acc,
                                    examples_per_sec, sec_per_batch, lr))
                summary_writer.add_summary(summary_str, step)
            else:
                _ = sess.run([train_op])
            # Save the model checkpoint periodically and do eval.
            if step % FLAGS.save_step == 0 or (step + 1) // num_per_epoch == FLAGS.max_epoch:
                checkpoint_path = os.path.join(FLAGS.train_dir,
                                               '{}_model.ckpt'.format(FLAGS.model_name))
                saver.save(sess, checkpoint_path, global_step=step)
                print ("save model at {}".format(step))
            step += 1
            epoch = step * num_examples_per_step // num_per_epoch

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
