from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import add_arg_scope
import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS


# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs]) as sc:
        squeeze = slim.conv2d(inputs, squeeze_depth, [1, 1],
                              scope='squeeze')
        outputs = expand(squeeze, expand_depth)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)

def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], axis=3)



def squeezenet_inference(inputs, is_training, keep_prob):
    nets = slim.conv2d(inputs, 64,
                       [3, 3], scope='conv1')
    nets = slim.max_pool2d(nets, [3, 3], padding='SAME', scope='pool1')  # 56*48*64

    nets = fire_module(nets, 16, 64, scope='fire2')

    nets = fire_module(nets, 16, 64, scope='fire3')

    nets = slim.max_pool2d(nets, [3, 3], padding='SAME', scope='pool1')  # 28*24*128

    nets = fire_module(nets, 32, 128, scope='fire4')

    nets = fire_module(nets, 32, 128, scope='fire5')

    nets = slim.max_pool2d(nets, [3, 3], padding='SAME', scope='pool5')  # 14*12*256

    nets = fire_module(nets, 48, 192, scope='fire6')

    nets = fire_module(nets, 48, 192, scope='fire7')

    nets = slim.max_pool2d(nets, [3, 3], padding='SAME', scope='pool6')  # 7*6*384

    nets = fire_module(nets, 64, 256, scope='fire8')

    nets = fire_module(nets, 64, 256, scope='fire9')  # 7*6*512

    nets = slim.dropout(nets, keep_prob, is_training=is_training, scope='dropout9')

    nets = slim.avg_pool2d(nets, [7, 6], scope='pool9')  # 1*1*512

    return nets


def squeezenet(inputs,
               num_classes=1000,
               is_training=True,
               keep_prob=0.5,
               spatial_squeeze=True,
               scope='squeeze'):
    """
    squeezenetv1.1
    """
    with tf.name_scope(scope, 'squeeze', [inputs]) as sc:
        end_points_collection = sc + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d,
                             slim.avg_pool2d, fire_module],
                            outputs_collections=end_points_collection):
            nets = squeezenet_inference(inputs, is_training, keep_prob)
            nets = slim.conv2d(nets, num_classes, [1, 1],
                               activation_fn=None,
                               normalizer_fn=None,
                               scope='logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                nets = tf.squeeze(nets, [1, 2], name='logits/squeezed')
            return nets, end_points


def inference(images, num_classes, is_training=True, scope='squeeze'):
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
            logits, endpoints = squeezenet(
                images,
                num_classes=num_classes,
                keep_prob=0.5,
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