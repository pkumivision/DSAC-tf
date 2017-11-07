from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

slim = tf.contrib.slim

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

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
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points

def objNet(inputs,
           num_classes=3,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='objNet'):
  with tf.variable_scope(scope, 'objNet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      # 42x42
      net = slim.conv2d(inputs, 64, [3, 3], stride=1, padding='VALID', scope='conv1')
      net = slim.conv2d(net, 64, [3, 3], stride=2, padding='SAME', scope='conv2')
      # 20x20
      net = slim.conv2d(net, 128, [3, 3], stride=1, padding='SAME', scope='conv3')
      net = slim.conv2d(net, 128, [3, 3], stride=2, padding='SAME', scope='conv4')
      # 10x10
      net = slim.conv2d(net, 256, [3, 3], stride=1, padding='SAME', scope='conv5')
      net = slim.conv2d(net, 256, [3, 3], stride=1, padding='SAME', scope='conv6')
      net = slim.conv2d(net, 256, [3, 3], stride=2, padding='SAME', scope='conv7')
      # 5x5
      net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='conv8')
      net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='conv9')
      net = slim.conv2d(net, 512, [3, 3], stride=2, padding='VALID', scope='conv10')
      # 2x2
      net = slim.conv2d(net, 4096, [2, 2], stride=1, padding='VALID', scope='fc11')
      net = slim.conv2d(net, 4096, [1, 1], stride=1, padding='VALID', scope='fc12')
      net = slim.conv2d(net, num_classes, [1, 1], stride=1, padding='VALID', scope='fc13',
                        activation_fn=None, normalizer_fn=None)

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc13/squeezed')
        end_points[sc.name + '/fc13'] = net
      return net, end_points

def scoreNet(inputs,
           num_classes=3,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='scoreNet'):
  with tf.variable_scope(scope, 'scoreNet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      # 40x40
      net = slim.conv2d(inputs, 32, [3, 3], stride=1, padding='SAME', scope='conv1')
      net = slim.conv2d(net, 32, [3, 3], stride=2, padding='SAME', scope='conv2')
      # 20x20
      net = slim.conv2d(net, 64, [3, 3], stride=1, padding='SAME', scope='conv3')
      net = slim.conv2d(net, 64, [3, 3], stride=2, padding='SAME', scope='conv4')
      # 10x10
      net = slim.conv2d(net, 128, [3, 3], stride=1, padding='SAME', scope='conv5')
      net = slim.conv2d(net, 128, [3, 3], stride=2, padding='SAME', scope='conv6')
      # 5x5
      net = slim.conv2d(net, 256, [3, 3], stride=1, padding='SAME', scope='conv7')
      net = slim.conv2d(net, 256, [3, 3], stride=2, padding='VALID', scope='conv8')
      # 2x2
      net = slim.conv2d(net, 512, [3, 3], stride=1, padding='SAME', scope='conv9')
      net = slim.conv2d(net, 512, [3, 3], stride=2, padding='SAME', scope='conv10')
      # 1x1
      net = slim.conv2d(net, 1024, [1, 1], stride=1, padding='VALID', scope='fc11')
      net = slim.conv2d(net, 1024, [1, 1], stride=1, padding='VALID', scope='fc12')
      net = slim.conv2d(net, num_classes, [1, 1], stride=1, padding='VALID', scope='fc13',
                        activation_fn=None, normalizer_fn=None)

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc13/squeezed')
        end_points[sc.name + '/fc13'] = net
      return net, end_points
