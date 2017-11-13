import tensorflow as tf
import time
import numpy as np
import math
from multiprocessing import Pool
import cv2
import os
from os import path

from utils import getInfo, stochasticSubSample, getDiffMap, calcDistance, our2cv, getCoordImg
import properties

class scoreGenerator(object):
    def __init__(self, opt, objLearner, sess):
            self.opt = opt
            self.ol = objLearner
            self.sess = sess
            self.rgb_paths = []
            self.pose_paths = []
            with open(opt.list) as f:
                for line in f.readlines():
                    rgb, pose = line.strip().split(' ')
                    self.rgb_paths.append(rgb)
                    self.pose_paths.append(pose)
            self.indexs = np.arange(len(self.rgb_paths))
            self.total = len(self.rgb_paths) * self.opt.training_hyps
            self._new_epoch()

    def _new_epoch(self):
        start_time = time.time()
        np.random.shuffle(self.indexs)
        self.data = []
        self.scores = []
        total_correct = 0.0
        for i in xrange(self.opt.training_images):
            curi = self.indexs[i]
            rgb = cv2.imread(path.join(self.opt.dataset_dir, self.rgb_paths[curi]))
            pose = getInfo(path.join(self.opt.dataset_dir, self.pose_paths[curi]))
            sampling = stochasticSubSample(self.opt.img_height, self.opt.img_width, self.opt.obj_size, self.opt.input_size)
            estObj = getCoordImg(rgb, sampling, self.sess, self.ol, self.opt)
            estObj = estObj.reshape(-1,3)

            for h in xrange(self.opt.training_hyps):
                diffMap, score, correct = createScore(pose, estObj, sampling)
                self.data.append(diffMap.reshape(40,40,1))
                self.scores.append(score.reshape(1))
                total_correct += correct

        self.step = 0
        if self.opt.time_info:
            print "Generated {} patches ({:2f}% correct) in {}s".format(len(self.data), total_correct/len(self.data)*100.0, time.time()-start_time)

    def _next(self):
        if self.step == self.opt.training_images*self.opt.training_hyps:
            self._new_epoch()
        data = self.data[self.step]
        score = self.scores[self.step]
        self.step += 1
        return data, score

    def next_batch(self):
        data = []
        label = []
        for i in xrange(self.opt.batch_size):
            d, l = self._next()
            data.append(d)
            label.append(l)

        return data, label

def getRandHyp(guassRot, gaussTrans):
    trans = np.random.normal(0, gaussTrans, size=3)
    rotAxis = np.random.normal(0, 1, size=3)
    rotAxis = rotAxis / np.linalg.norm(rotAxis)
    rotAxis = rotAxis * np.random.normal(0, guassRot) * math.pi / 180.0
    rot,_ = cv2.Rodrigues(rotAxis)
    transform = np.eye(4)
    transform[:3,:3] = rot
    transform[:3,3] = trans
    return transform

def createScore(poseGT, estObj, sampling):
    driftLevel = np.random.randint(2)
    if driftLevel == 0:
        poseNoise = np.matmul(poseGT, getRandHyp(2, 2))
    else:
        poseNoise = np.matmul(poseGT, getRandHyp(10, 100))
    poseNoise_cv = our2cv(poseNoise)
    diffMap = getDiffMap(poseNoise_cv, estObj, sampling, properties.getCamMat())
    angularDistance, translationDistance = calcDistance(poseGT, poseNoise)
#    print angularDistance, translationDistance
    correct = 0
    if angularDistance<5 and translationDistance<50:
        correct = 1
    score = 10 * np.max((angularDistance, translationDistance/10))
    return diffMap, score, correct
