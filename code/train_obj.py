from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from objLearner import objLearner
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "../image", "Dataset directory")
flags.DEFINE_string("list", "../list/train.list", "Image list directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("loss", "L2", "loss type")
flags.DEFINE_string("gpu_no", "0", "gpu number")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of for adam")
flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("img_height", 480, "Image height")
flags.DEFINE_integer("img_width", 640, "Image width")
flags.DEFINE_integer("lr_step", 50000, "Learning rate decay step")
flags.DEFINE_integer("summary_freq", 1000, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 100000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")

flags.DEFINE_integer("input_size", 42, "RGB patch size")
flags.DEFINE_integer("channel", 3, "Image channel")
flags.DEFINE_integer("training_images", 100, "number of training images randonly chosen in each training round")
flags.DEFINE_integer("training_patches", 512, "number of patches extracted from each training image")
flags.DEFINE_integer("max_steps", 300000, "Maximum number of training iterations")
flags.DEFINE_integer("batch_size", 64, "training batch size")

flags.DEFINE_boolean("time_info", False, "show some time information")

FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    ol = objLearner(FLAGS)
    ol.train()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no
    tf.app.run()
