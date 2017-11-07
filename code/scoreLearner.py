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

from objLearner import objLearner
from scoreGenerator import scoreGenerator

class ScoreLearner(object):
    def __init__(self, opt):
        self.opt = opt

    def compute_loss(self, preds, labels):
        with tf.name_scope('compute_loss'):
            loss = tf.reduce_mean(tf.abs(preds-labels))
            return loss

    def collect_summaries(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
#        for var in tf.trainable_variables():
#            tf.summary.histogram(var.op.name+'/values', var)
        for grad, var in self.grads_and_vars:
            tf.summary.scalar(var.op.name+'/gradients', tf.reduce_mean(grad))
        #tf.summary.image('image', self.input)

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

    def build_graph(self, mode):
        self.ol = objLearner(self.opt)
        self.ol.build_graph()

        self.input = tf.placeholder(dtype=tf.uint8, shape=(self.opt.batch_size, self.opt.obj_size, self.opt.obj_size, self.opt.channel))
        self.labels = tf.placeholder(dtype=tf.float32, shape=(self.opt.batch_size))
        input_process = tf.cast(self.input, tf.float32) - 45.0
        self.preds, self.end_points = scoreNet(input_process)

    def train(self):
        self.build_graph('train')
        self.loss = self.compute_loss(self.preds, self.labels)

        self.learning_rate = tf.Variable(0.0, trainable=False)
        new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        lr_update = tf.assign(self.learning_rate, new_lr)

        self.get_train_op()
        self.collect_summaries()

        obj_parameters = []
        score_parameters = []
        with tf.name_scope("parameter_count"):
            for v in tf.trainable_variables():
                if v.name.startswith('score'):
                    score_parameters.append(v)
                elif v.name.startswith('obj'):
                    obj_parameters.append(v)
                else:
                    print 'UNKNOWN PARAMETER:', v.name
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in score_parameters])

        self.saver = tf.train.Saver([var for var in score_parameters] + \
                                    [self.global_step])
        obj_saver = tf.train.Saver([var for var in obj_parameters])

        sv = tf.train.Supervisor(logdir=self.opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        with sv.managed_session() as sess:
            print 'All variables:'
            for var in tf.trainable_variables():
                print var.name,
            print 'Trainable variables:'
            for var in score_parameters:
                print var.name,
            print
            print 'parameter count =', sess.run(parameter_count)
            print 'Loading ObjNet...'
            obj_saver.restore(sess, self.opt.obj_model)
            print 'Load ObjNet model successfully'
            if self.opt.continue_train:
                print 'Resume training from previous checkpoint'
                checkpoint = tf.train.latest_checkpoint(self.opt.checkpoint_dir)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            acc_loss = 0
            lr_value = self.opt.learning_rate
            data_generator = scoreGenerator(self.opt, sess)
            self.total = data_generator.total
            self.opt.steps_per_epoch = int(self.total//self.opt.batch_size)

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

                if gs % self.opt.lr_step == 0:
                    lr_value = self.opt.learning_rate * (self.opt.learning_rate_decay ** int(gs/self.opt.lr_step))
                    print '[*] Learning Rate Update to', lr_value

                if step % self.opt.save_latest_freq == 0:
                    self.save(sess, self.opt.checkpoint_dir, gs)
            self.save(sess, self.opt.checkpoint_dir, gs+1)
        print 'optimize done'

    def test(self):
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
