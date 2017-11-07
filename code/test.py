from __future__ import division
import os
import math
import pprint
import tensorflow as tf
import numpy as np

from objLearner import objLearner

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "../image", "Dataset directory")
flags.DEFINE_string("list", "../list/train.list", "Image list directory")
flags.DEFINE_string("gpu_no", "0", "gpu number")
flags.DEFINE_integer("img_height", 480, "Image height")
flags.DEFINE_integer("img_width", 640, "Image width")
flags.DEFINE_integer("channel", 3, "Image channel")
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_string("output_file", None, "Output file")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")

flags.DEFINE_integer("input_size", 42, "RGB patch size")

FLAGS = flags.FLAGS

def main(_):

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    ol = objLearner(FLAGS)
    ol.test()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no
    tf.app.run()
