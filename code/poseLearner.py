from __future__ import division
import os
from os import path
import time
import math
import random
import numpy as np
import tensorflow as tf
from nets import *
import cv2

from dataset import DataGenerator

class poseLearner(object):
    def __init__(self, opt):
        self.opt = opt

    def get_input(self, mode='train'):
        print '[*] get input from list'
        with tf.name_scope('data_loading'):
            seed = random.randint(0, 2**31-1)

            # Load the list of training files into queues
            file_list, self.total = self.get_file_list(self.opt.image_list_dir)
            input_queue = tf.train.string_input_producer(
                file_list,
                seed=seed,
                shuffle=mode=='train')

            # Load images and labels
            reader = tf.TextLineReader()
            _, line = reader.read(input_queue)
            record_defaults = [['null'], ['null'], ['null'], ['0']]
            pet, ct1, ct2, label = tf.decode_csv(line, record_defaults, ' ')
            if len(self.opt.dataset_dir)>0 and not self.opt.dataset_dir.endswith('/'):
                self.opt.dataset_dir += '/'
            image_name = [pet, ct1, ct2]
            images = []
            for name in image_name:
                image_content = tf.read_file(tf.string_join([self.opt.dataset_dir, name]))
                image = tf.image.decode_png(image_content,dtype=tf.uint16)
                image.set_shape([self.opt.img_height,self.opt.img_width,1])
                images.append(image)

            image_name = tf.stack(image_name)
            image_name.set_shape([3])
            image = tf.concat(images, 2)
            label = tf.string_to_number(label, out_type=tf.float32)
            label.set_shape([])

            min_after_dequeue = self.opt.shuffle_len
            capacity = min_after_dequeue + (self.opt.reader_num + 1) * self.opt.batch_size
            image_batch, label_batch, path_batch = tf.train.shuffle_batch(
                                                    [image, label, image_name],
                                                    batch_size=self.opt.batch_size,
                                                    num_threads=self.opt.reader_num,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
            return image_batch, label_batch, path_batch

    def get_input_tfrecord(self, mode='train'):
        print '[*] get input from tfrecord'
        with tf.name_scope('data_loading'):
            seed = random.randint(0, 2**31-1)

            # Load the list of training files into queues
            file_list = self.get_tfrecord_list(self.opt.tfrecord_dir)
            #print file_list
            self.total = self.opt.total
            input_queue = tf.train.string_input_producer(
                file_list,
                seed=seed,
                shuffle=mode=='train')

            # Load images and labels
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(input_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                  'image/height': tf.FixedLenFeature([], tf.int64),
                  'image/width': tf.FixedLenFeature([], tf.int64),
                  'image/colorspace': tf.FixedLenFeature([], tf.string),
                  'image/channels': tf.FixedLenFeature([], tf.int64),
                  'image/label': tf.FixedLenFeature([], tf.int64),
                  'image/format': tf.FixedLenFeature([], tf.string),
                  'image/filename': tf.FixedLenFeature([], tf.string),
                  'image/encoded': tf.FixedLenFeature([], tf.string)
                })
            image = tf.decode_raw(features['image/encoded'], tf.float32)
            label = tf.cast(features['image/label'], tf.float32)
            image_name = features['image/filename']
            height = tf.cast(features['image/height'], tf.int32)
            width = tf.cast(features['image/width'], tf.int32)
            channels = tf.cast(features['image/channels'], tf.int32)

            image_shape = tf.stack([224,224,3])
            image = tf.reshape(image, image_shape)
            label.set_shape([])

            min_after_dequeue = self.opt.shuffle_len
            capacity = min_after_dequeue + (self.opt.reader_num + 1) * self.opt.batch_size
            image_name.set_shape([])
            image_batch, label_batch, path_batch = tf.train.shuffle_batch(
                                                    [image, label, image_name],
                                                    batch_size=self.opt.batch_size,
                                                    num_threads=self.opt.reader_num,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
            return image_batch, label_batch, path_batch

    def compute_loss(self, preds, labels):
        with tf.name_scope('compute_loss'):
            if self.opt.loss == "L1":
                loss = tf.reduce_mean(tf.abs(preds-labels))
            else:
                loss = tf.reduce_mean(tf.norm(preds-labels, axis=1))
            return loss

    def collect_summaries(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
#        for var in tf.trainable_variables():
#            tf.summary.histogram(var.op.name+'/values', var)
        for grad, var in self.grads_and_vars:
            tf.summary.scalar(var.op.name+'/gradients', tf.reduce_mean(grad))
        tf.summary.image('image', self.input)

    def get_train_op(self):
        train_vars = [var for var in tf.trainable_variables()]
        optim = tf.train.AdamOptimizer(self.learning_rate, self.opt.beta1)
        self.grads_and_vars = optim.compute_gradients(self.loss,
                                                          var_list=train_vars)
        self.train_op = optim.apply_gradients(self.grads_and_vars)
        self.global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False)
        self.incr_global_step = tf.assign(self.global_step,
                                          self.global_step+1)

    def isPretrainVar(self, s):
        return self.opt.pretrain_model is not None and s.startswith('vgg_16') and not 'fc8' in s and not 'my' in s

    def build_graph(self, mode):
        self.input = tf.placeholder(dtype=tf.uint8, shape=(self.opt.batch_size, self.opt.input_size, self.opt.input_size, self.opt.channel))
        self.labels = tf.placeholder(dtype=tf.float32, shape=(self.opt.batch_size, 3))
        input_process = tf.cast(self.input, tf.float32) - 127.0
        self.preds, self.end_points = objNet(input_process)

    def train(self):
        self.build_graph('train')
        self.total = self.opt.training_images * self.opt.training_patches
        self.opt.steps_per_epoch = int(self.total//self.opt.batch_size)
        self.loss = self.compute_loss(self.preds, self.labels)

        self.learning_rate = tf.Variable(0.0, trainable=False)
        new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        lr_update = tf.assign(self.learning_rate, new_lr)

        self.get_train_op()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.trainable_variables()] + \
                                    [self.global_step])
        pretrain_vars = []
        for var in tf.trainable_variables():
            if self.isPretrainVar(var.name):
                pretrain_vars.append(var)
        if self.opt.pretrain_model is not None:
            self.finetune_saver = tf.train.Saver(pretrain_vars)
        sv = tf.train.Supervisor(logdir=self.opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        with sv.managed_session() as sess:
            print 'Trainable variables:'
            for var in tf.trainable_variables():
                print var.name,
            print
            print 'parameter count =', sess.run(parameter_count)
            if self.opt.continue_train:
                print 'Resume training from previous checkpoint'
                checkpoint = tf.train.latest_checkpoint(self.opt.checkpoint_dir)
                self.saver.restore(sess, checkpoint)
            elif self.opt.pretrain_model is not None:
                self.finetune_saver.restore(sess=sess, save_path=self.opt.pretrain_model)
                print 'finetune from', self.opt.pretrain_model
                print 'pretrain weights:'
                print pretrain_vars
            start_time = time.time()
            acc_loss = 0
            lr_value = self.opt.learning_rate
            data_generator = DataGenerator(self.opt)

            for step in xrange(1, self.opt.max_steps):
                fetches = {
                    'train': self.train_op,
                    'global_step': self.global_step,
                    'incr_global_step': self.incr_global_step,
                    'loss': self.loss,
                    'lr': lr_update
                }

                if step % self.opt.summary_freq == 0:
                    fetches['summary'] = sv.summary_op
                   # fetches['label'] = self.labels
                   # fetches['name'] = self.image_paths
                   # fetches['preds'] = self.preds

                feed_input, feed_label = data_generator.next_batch()

                results = sess.run(fetches, feed_dict={new_lr:lr_value,
                                                        self.input:feed_input,
                                                        self.labels:feed_label})
                gs = results['global_step']
                acc_loss += results['loss']

                if step % self.opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results['summary'], gs)

                    train_epoch = int(math.ceil(gs / self.opt.steps_per_epoch))
                    train_step = gs - (train_epoch - 1) * self.opt.steps_per_epoch

                    avg_loss = acc_loss/self.opt.summary_freq

                    print 'Epoch: {:2d} {:5d}/{:5d}  time: {:4.4f}s/iter  loss: {:.3f}' \
                            .format(train_epoch, train_step, self.opt.steps_per_epoch, \
                                    (time.time()-start_time)/self.opt.summary_freq, \
                                    avg_loss)

                    start_time = time.time()
                    acc_loss = 0

                   # print results['label']
                   # print self.preprocess_label(results['label'],self.opt.label_alpha,self.opt.label_beta)
                   # print results['preds']

                if gs % self.opt.lr_step == 0:
                    lr_value = self.opt.learning_rate * (self.opt.learning_rate_decay ** int(gs/self.opt.lr_step))
                    print '[*] Learning Rate Update to', lr_value

                if step % self.opt.save_latest_freq == 0:
                    self.save(sess, self.opt.checkpoint_dir, gs)
                #if step % self.opt.steps_per_epoch == 0:
                #    self.save(sess, self.opt.checkpoint_dir, gs)
            self.save(sess, self.opt.checkpoint_dir, gs+1)
        print 'optimize done'

    def test(self):
        def sigmoid(x):
            return 1.0/(1.0+np.exp(-x))

        self.build_graph('test')
        saver = tf.train.Saver([var for var in tf.trainable_variables()])
        with open(self.opt.output_file, 'w') as out:
            with tf.Session() as sess:
                print 'Restore from', self.opt.ckpt_file
                saver.restore(sess, self.opt.ckpt_file)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                start_time = time.time()
                total = int(math.ceil(self.total/self.opt.batch_size))

                t = f = 0.0

                for itr in xrange(1, total+1):
                    fetches = {'preds' : self.preds,
                               'image_path' : self.image_paths,
                               'labels': self.labels}
                    results = sess.run(fetches)
                    print '\r{}/{}  time={}/img'.format(itr, total, (time.time()-start_time)/itr/self.opt.batch_size),
                    preds = results['preds']
                    paths = results['image_path']
                    labels = results['labels']
                    for i in xrange(len(preds)):
                        pl = sigmoid(preds[i][0])
                        print >> out, paths[i], '{},{}'.format(pl, labels[i])
                        if labels[i] == 1 and pl>0.5: t += 1
                        elif labels[i] == 0 and pl<=0.5: t += 1
                        else:
                            f += 1
                            print paths[i], pl, labels[i]

                print 'accuracy:', t/(t+f)

                coord.request_stop()
                coord.join(threads)

    def preprocess_image(self, image):
        if image.dtype == tf.uint8:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        else:
            image = image / 255
        return image * 2. - 1.

    def deprocess_image(self, image, dtype=tf.uint8):
        image = (image + 1.) / 2.
        return tf.image.convert_image_dtype(image, dtype=dtype)

    def get_file_list(self, root_dir):
        total = 0
        file_list = []
        files = os.listdir(root_dir)

        for f in files:
            file_path = path.join(root_dir, f)
            file_list.append(file_path)
            with open(file_path) as i:
                total += len(i.readlines())
        return file_list, total

    def get_tfrecord_list(self, root_dir):
        file_list = []
        files = os.listdir(root_dir)
        for f in files:
            file_list.append(path.join(root_dir, f))
        return file_list

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print '[*] Saving checkpoint to {}-{}'.format(checkpoint_dir, step)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
