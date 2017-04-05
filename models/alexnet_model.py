"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.framework.ops import name_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
import tensorflow as tf

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

FLAGS = tf.app.flags.FLAGS

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997


def alexnet_v2_arg_scope(weight_decay=0.0005):
    with arg_scope(
            [layers.conv2d, layers_lib.fully_connected],
            activation_fn=nn_ops.relu,
            biases_initializer=init_ops.constant_initializer(0.1),
            weights_regularizer=regularizers.l2_regularizer(weight_decay)):
        with arg_scope([layers.conv2d], padding='SAME'):
            with arg_scope([layers_lib.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2'):
    """AlexNet version 2.

      Described in: http://arxiv.org/pdf/1404.5997v2.pdf
      Parameters from:
      github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
      layers-imagenet-1gpu.cfg

    Note: All the fully_connected layers have been transformed to conv2d layers.
            To use in classification mode, resize input to 224x224. To use in fully
            convolutional mode, set spatial_squeeze to false.
            The LRN layers have been removed and change the initializers from
            random_normal_initializer to xavier_initializer.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
          layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
          outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.

    Returns:
        the last op containing the log predictions and end_points dict.
    """
    with name_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with arg_scope(
                [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                outputs_collections=[end_points_collection]):
            net = layers.conv2d(
                inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = layers.conv2d(net, 192, [5, 5], scope='conv2')
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = layers.conv2d(net, 384, [3, 3], scope='conv3')
            net = layers.conv2d(net, 384, [3, 3], scope='conv4')
            net = layers.conv2d(net, 256, [3, 3], scope='conv5')
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected layers.
            with arg_scope(
                    [layers.conv2d],
                    weights_initializer=trunc_normal(0.005),
                    biases_initializer=init_ops.constant_initializer(0.1)):
                net = layers.conv2d(net, 4096, [2, 1], padding='VALID', scope='fc6')
                net = layers_lib.dropout(
                    net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
                net = layers_lib.dropout(
                    net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = layers.conv2d(
                    net,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    scope='logits')

            # Convert end_points_collection into a end_point dict.
            end_points = utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points


def inference(images, num_classes, is_training=True, scope='alexnet_v2'):
    """Build Inception v3 model architecture.

     See here for reference: http://arxiv.org/abs/1512.00567

    Args:
        images: Images returned from inputs() or distorted_inputs().
        num_classes: number of classes
        for_training: If set to `True`, build the inference model for training.
        Kernels that operate differently for inference during training
        e.g. dropout, are appropriately configured.
        restore_logits: whether or not the logits layers should be restored.
        Useful for fine-tuning a model with different num_classes.
        scope: optional prefix string identifying the ImageNet tower.

    Returns:
        Logits. 2-D float Tensor.
        Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # calculate moving average or using exist one
        'is_training': is_training
    }
    # Set weight_decay for weights in Conv and FC layers.
    with arg_scope([layers.conv2d, layers.fully_connected],
                   activation_fn=nn_ops.relu,
                   biases_initializer=init_ops.constant_initializer(0.1),
                   weights_regularizer=regularizers.l2_regularizer(FLAGS.weight_decay)):
        with arg_scope([layers.conv2d],
                       normalizer_fn=layers.batch_norm,
                       normalizer_params=batch_norm_params):
            logits, endpoints = alexnet_v2(
                images,
                num_classes=num_classes,
                dropout_keep_prob=0.8,
                is_training=is_training,
                scope=scope
            )

    # Add summaries for viewing model statistics on TensorBoard.
    _activation_summaries(endpoints)

    return logits


def loss(logits, labels):
    """Adds all losses for the model.

  Note the final loss is not returned. Instead, the list of losses are collected
  by slim.losses. The losses are accumulated in tower_loss() and summed to
  calculate the total loss.

  Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  """
    # Reshape the labels into a dense Tensor of
    # shape [FLAGS.batch_size, num_classes].
    with tf.name_scope("train_loss"):
        # cal softmax loss
        num_classes = logits[0].get_shape()[-1].value
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)
        softmax_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                       logits=logits,
                                                       label_smoothing=FLAGS.label_smoothing,
                                                       scope='softmax_loss')
        tf.summary.scalar('softmax_loss', softmax_loss)

        # cal regular loss
        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar('train_regular_loss', regularization_loss)

        # cal total loss
        total_loss = softmax_loss + regularization_loss
        tf.summary.scalar('train_total_loss', total_loss)
        return total_loss


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    """
    # session. This helps the clarity of presentation on tensorboard.
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            _activation_summary(act)
