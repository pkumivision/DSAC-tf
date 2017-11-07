import tensorflow as tf
import os
from os import path
import cv2

from rhLearner import rhLearner

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "../image", "Dataset directory")
flags.DEFINE_string("image_list_dir", "../list/test", "Image list file")
flags.DEFINE_string("tfrecord_dir", "./data/30/tfrecord", "TFRecord directory")
flags.DEFINE_string("output_dir", "test", "Test output directory")
flags.DEFINE_string("gpu_no", "0", "gpu number")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("reader_num", 2, "The number of of reader")
flags.DEFINE_integer("img_height", 224, "Image height")
flags.DEFINE_integer("img_width", 224, "Image width")
flags.DEFINE_integer("img_channel", 3, "Image channel")
flags.DEFINE_integer("shuffle_len", 100, "Shuffle length")
flags.DEFINE_integer("total", 100, "Image total, only used in tfrecord mode")
flags.DEFINE_boolean("using_tfrecord", False, "Reading data from TFRecord")
FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no

with tf.Session() as sess:
    rh = rhLearner(FLAGS)
    if FLAGS.using_tfrecord:
        images, labels, names = rh.get_input_tfrecord()
    else:
        images, labels, names = rh.get_input()
    print 'total:', rh.total

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if not path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    for i in xrange(10):
        ls, imgs, ns = sess.run([labels,images,names])
        if FLAGS.using_tfrecord:
            imgs = (imgs + 1) / 2 * 255
        print ns
        for j in xrange(len(ls)):
            for k in xrange(3):
                img_path = path.join(FLAGS.output_dir, '{}_{}_{}'.format(i, int(ls[j]), ns[j][k].replace('/','_')))
                cv2.imwrite(img_path, imgs[j][:,:,k])

    coord.request_stop()
    coord.join(threads)
