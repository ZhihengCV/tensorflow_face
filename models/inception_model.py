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
"""Build the Inception v3 network on ImageNet data set.

The Inception v3 architecture is described in http://arxiv.org/abs/1512.00567

Summary of available functions:
 inference: Compute inference on the model inputs to make a prediction
 loss: Compute the loss of the prediction with respect to the labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS


# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997


def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 scope='inception_v3'):
    """Latest Inception from http://arxiv.org/abs/1512.00567.

        "Rethinking the Inception Architecture for Computer Vision"

        Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
        Zbigniew Wojna

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        dropout_keep_prob: dropout keep_prob.
        num_classes: number of predicted classes.
        is_training: whether is training or not.
        restore_logits: whether or not the logits layers should be restored.
        Useful for fine-tuning a model with different num_classes.
        scope: Optional scope for name_scope.

    Returns:
        a list containing 'logits', 'aux_logits' Tensors.
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}
    with tf.name_scope(scope, 'inception_v3', [inputs]):
        with slim.arg_scope([slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                stride=1, padding='VALID'):
                # 299 x 299 x 3
                end_points['conv0'] = slim.conv2d(inputs, 32, [3, 3], stride=2,
                                                  scope='conv0')
                # 149 x 149 x 32
                end_points['conv1'] = slim.conv2d(end_points['conv0'], 32, [3, 3],
                                                  scope='conv1')
                # 147 x 147 x 32
                end_points['conv2'] = slim.conv2d(end_points['conv1'], 64, [3, 3],
                                                  padding='SAME', scope='conv2')
                # 147 x 147 x 64
                end_points['pool1'] = slim.max_pool2d(end_points['conv2'], [3, 3],
                                                    stride=2, scope='pool1')
                # 73 x 73 x 64
                end_points['conv3'] = slim.conv2d(end_points['pool1'], 80, [1, 1],
                                                  scope='conv3')
                # 73 x 73 x 80.
                end_points['conv4'] = slim.conv2d(end_points['conv3'], 192, [3, 3],
                                                  scope='conv4')
                # 71 x 71 x 192.
                end_points['pool2'] = slim.max_pool2d(end_points['conv4'], [3, 3],
                                                    stride=2, scope='pool2')
                # 35 x 35 x 192.
                net = end_points['pool2']
            # Inception blocks
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # mixed: 35 x 35 x 256.
                with tf.variable_scope('mixed_35x35x256a'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = slim.conv2d(net, 64, [1, 1])
                    with tf.variable_scope('branch5x5'):
                        branch5x5 = slim.conv2d(net, 48, [1, 1])
                        branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = slim.conv2d(net, 64, [1, 1])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.avg_pool2d(net, [3, 3])
                        branch_pool = slim.conv2d(branch_pool, 32, [1, 1])
                    net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)
                    end_points['mixed_35x35x256a'] = net
                # mixed_1: 35 x 35 x 288.
                with tf.variable_scope('mixed_35x35x288a'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = slim.conv2d(net, 64, [1, 1])
                    with tf.variable_scope('branch5x5'):
                        branch5x5 = slim.conv2d(net, 48, [1, 1])
                        branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = slim.conv2d(net, 64, [1, 1])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.avg_pool2d(net, [3, 3])
                        branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
                    net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)
                    end_points['mixed_35x35x288a'] = net
                # mixed_2: 35 x 35 x 288.
                with tf.variable_scope('mixed_35x35x288b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = slim.conv2d(net, 64, [1, 1])
                    with tf.variable_scope('branch5x5'):
                        branch5x5 = slim.conv2d(net, 48, [1, 1])
                        branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = slim.conv2d(net, 64, [1, 1])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.avg_pool2d(net, [3, 3])
                        branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
                    net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)
                    end_points['mixed_35x35x288b'] = net
                # mixed_3: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768a'):
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = slim.conv2d(net, 64, [1, 1])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3],
                                                   stride=2, padding='VALID')
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
                    net = tf.concat([branch3x3, branch3x3dbl, branch_pool], 3)
                    end_points['mixed_17x17x768a'] = net
                # mixed4: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = slim.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = slim.conv2d(net, 128, [1, 1])
                        branch7x7 = slim.conv2d(branch7x7, 128, [1, 7])
                        branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = slim.conv2d(net, 128, [1, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [1, 7])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.avg_pool2d(net, [3, 3])
                        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
                    end_points['mixed_17x17x768b'] = net
                # mixed_5: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768c'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = slim.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = slim.conv2d(net, 160, [1, 1])
                        branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
                        branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = slim.conv2d(net, 160, [1, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.avg_pool2d(net, [3, 3])
                        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
                    end_points['mixed_17x17x768c'] = net
                # mixed_6: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768d'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = slim.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = slim.conv2d(net, 160, [1, 1])
                        branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
                        branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = slim.conv2d(net, 160, [1, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.avg_pool2d(net, [3, 3])
                        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
                    end_points['mixed_17x17x768d'] = net
                # mixed_7: 17 x 17 x 768.
                with tf.variable_scope('mixed_17x17x768e'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = slim.conv2d(net, 192, [1, 1])
                    with tf.variable_scope('branch7x7'):
                        branch7x7 = slim.conv2d(net, 192, [1, 1])
                        branch7x7 = slim.conv2d(branch7x7, 192, [1, 7])
                        branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
                    with tf.variable_scope('branch7x7dbl'):
                        branch7x7dbl = slim.conv2d(net, 192, [1, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
                        branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.avg_pool2d(net, [3, 3])
                        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
                    end_points['mixed_17x17x768e'] = net
                # Auxiliary Head logits
                aux_logits = tf.identity(end_points['mixed_17x17x768e'])
                with tf.variable_scope('bypath'):
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 4], stride=3,
                                               padding='VALID')
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='proj')
                    # Shape of feature map before the final layer.
                    shape = aux_logits.get_shape()
                    aux_logits = slim.conv2d(aux_logits, 768, shape[1:3],
                                             padding='VALID')
                    aux_logits = slim.flatten(aux_logits)
                    aux_logits = slim.fully_connected(aux_logits, num_classes, activation_fn=None,
                                                      normalizer_fn=None, scope='aux_logits')
                    end_points['aux_logits'] = aux_logits
                # mixed_8: 8 x 8 x 1280.
                # Note that the scope below is not changed to not void previous
                # checkpoints.
                # (TODO) Fix the scope when appropriate.
                with tf.variable_scope('mixed_17x17x1280a'):
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = slim.conv2d(net, 192, [1, 1])
                        branch3x3 = slim.conv2d(branch3x3, 320, [3, 3], stride=2,
                                                padding='VALID')
                    with tf.variable_scope('branch7x7x3'):
                        branch7x7x3 = slim.conv2d(net, 192, [1, 1])
                        branch7x7x3 = slim.conv2d(branch7x7x3, 192, [1, 7])
                        branch7x7x3 = slim.conv2d(branch7x7x3, 192, [7, 1])
                        branch7x7x3 = slim.conv2d(branch7x7x3, 192, [3, 3],
                                                  stride=2, padding='VALID')
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
                    net = tf.concat([branch3x3, branch7x7x3, branch_pool], 3)
                    end_points['mixed_17x17x1280a'] = net
                # mixed_9: 8 x 8 x 2048.
                with tf.variable_scope('mixed_8x8x2048a'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = slim.conv2d(net, 320, [1, 1])
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = slim.conv2d(net, 384, [1, 1])
                        branch3x3 = tf.concat([slim.conv2d(branch3x3, 384, [1, 3]),
                                               slim.conv2d(branch3x3, 384, [3, 1])], 3)
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = slim.conv2d(net, 448, [1, 1])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
                        branch3x3dbl = tf.concat([slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                                  slim.conv2d(branch3x3dbl, 384, [3, 1])], 3)
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.avg_pool2d(net, [3, 3])
                        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 3)
                    end_points['mixed_8x8x2048a'] = net
                # mixed_10: 8 x 8 x 2048.
                with tf.variable_scope('mixed_8x8x2048b'):
                    with tf.variable_scope('branch1x1'):
                        branch1x1 = slim.conv2d(net, 320, [1, 1])
                    with tf.variable_scope('branch3x3'):
                        branch3x3 = slim.conv2d(net, 384, [1, 1])
                        branch3x3 = tf.concat([slim.conv2d(branch3x3, 384, [1, 3]),
                                               slim.conv2d(branch3x3, 384, [3, 1])], 3)
                    with tf.variable_scope('branch3x3dbl'):
                        branch3x3dbl = slim.conv2d(net, 448, [1, 1])
                        branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
                        branch3x3dbl = tf.concat([slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                                  slim.conv2d(branch3x3dbl, 384, [3, 1])], 3)
                    with tf.variable_scope('branch_pool'):
                        branch_pool = slim.avg_pool2d(net, [3, 3])
                        branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                    net = tf.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 3)
                    end_points['mixed_8x8x2048b'] = net
                # Final pooling and prediction
                with tf.variable_scope('main_path'):
                    shape = net.get_shape()
                    net = slim.avg_pool2d(net, shape[1:3], padding='VALID', scope='pool')
                    # 1 x 1 x 2048
                    net = slim.dropout(net, dropout_keep_prob, scope='dropout')
                    net = slim.flatten(net, scope='flatten')
                    # 2048
                    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                                  normalizer_fn=None, scope='logits')
                    # 1000
                    end_points['logits'] = logits
                    end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
            return logits, end_points


def training_inference(images, num_classes, scope='inception_v3'):
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
        'is_training': True
    }
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            logits, endpoints = inception_v3(
                images,
                num_classes=num_classes,
                dropout_keep_prob=0.8,
                is_training=True,
                scope=scope
            )

    # Add summaries for viewing model statistics on TensorBoard.
    _activation_summaries(endpoints)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['aux_logits']

    return logits, auxiliary_logits


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
