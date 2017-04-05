from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS


# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997


def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
            return outputs

def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')

def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 3)


def densenet_block(inputs, layer_num, growth, bc_mode, scope, is_training, keep_prob):
    with tf.variable_scope(scope, 'block1', [inputs]):
        currents = inputs
        for idx in xrange(layer_num):
            if not bc_mode:
                new_feature = slim.conv2d(currents, growth,
                                          [3, 3], scope='conv_{:d}'.format(idx))
                new_feature = slim.dropout(new_feature, keep_prob=keep_prob,
                                           is_training=is_training,
                                           scope='dropout_{:d}'.format(idx))
            else:
                new_feature = slim.conv2d(currents, growth*4,
                                          [1, 1], scope='bottom_{:d}'.format(idx))
                new_feature = slim.dropout(new_feature, keep_prob=keep_prob,
                                           is_training=is_training,
                                           scope='dropout_b_{:d}'.format(idx))
                new_feature = slim.conv2d(new_feature, growth,
                                          [3, 3], scope='conv_{:d}'.format(idx))
                new_feature = slim.dropout(new_feature, keep_prob=keep_prob,
                                           is_training=is_training,
                                           scope='dropout_{:d}'.format(idx))
            currents = tf.concat([currents, new_feature], axis=3)
        return currents


def transition_block(inputs, reduction, scope, is_training, keep_prob):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    with tf.variable_scope(scope, 'trans1', [inputs]):
        # call composite function with 1x1 kernel
        out_features = int(int(inputs.get_shape()[-1]) * reduction)
        nets = slim.conv2d(inputs, out_features,
                           [1, 1], scope='conv')
        nets = slim.dropout(nets, keep_prob=keep_prob,
                            is_training=is_training,
                            scope='dropout')
        # run average pooling
        nets = slim.avg_pool2d(nets, [2, 2], scope='pool')
        return nets


def densenet_inference(inputs, is_training, keep_prob, growth_rate, reduction):

    first_output_fea = growth_rate * 2

    nets = slim.conv2d(inputs, first_output_fea,
                       [5, 5], scope='conv0')
    nets = slim.max_pool2d(nets, [3, 3], padding='SAME', scope='pool0')  # 56*48*64

    nets = densenet_block(nets, 6, growth_rate, True,
                          'block1', is_training, keep_prob)
    nets = transition_block(nets, reduction, 'trans1', is_training, keep_prob)  # 28*24*256

    nets = densenet_block(nets, 12, growth_rate, True,
                          'block2', is_training, keep_prob)
    nets = transition_block(nets, reduction, 'trans2', is_training, keep_prob)  # 14*12*640

    nets = densenet_block(nets, 24, growth_rate, True,
                          'block3', is_training, keep_prob)
    nets = transition_block(nets, reduction, 'trans3', is_training, keep_prob)  # 7*6*1408

    nets = densenet_block(nets, 16, growth_rate, True,
                          'block4', is_training, keep_prob)  # 7*6*1920
    nets = slim.avg_pool2d(nets, [7, 6], scope='pool4')  # 1*1*1920
    return nets


def densenet_a(inputs,
               num_classes=1000,
               is_training=True,
               keep_prob=0.2,
               growth_rate=32,
               reduction=0.6,
               spatial_squeeze=True,
               scope='densenet_121'):
    """
    Densenet 121-Layers version.
    """
    with tf.name_scope(scope, 'densenet_121', [inputs]) as sc:
        end_points_collection = sc + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d,
                             slim.avg_pool2d],
                            outputs_collections=end_points_collection):

            nets = densenet_inference(inputs, is_training, keep_prob, growth_rate, reduction)
            nets = slim.conv2d(nets, num_classes, [1, 1],
                               activation_fn=None,
                               normalizer_fn=None,
                               scope='logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                nets = tf.squeeze(nets, [1, 2], name='logits/squeezed')
            return nets, end_points


def inference(images, num_classes, is_training=True,scope='densenet_121'):
    """
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
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            logits, endpoints = densenet_a(
                images,
                num_classes=num_classes,
                keep_prob=0.2,
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

