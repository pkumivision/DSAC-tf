import tensorflow as tf
import time
import numpy as np
import math
from multiprocessing import Pool

from utils import getInfo, stochasticSubSample, getDiffMap, calcDistance, our2cv
from objLearner import objLearner
import properties

class ScoreGenerator(object):
	def __init__(self, opt, sess):
		self.opt = opt
		self.ol = objLearner(opt)
		self.sess = self.ol.predict_init(sess)
		self.rgb_paths = []
		self.pose_paths = []
		with open(opt.list) as f:
            for line in f.readlines():
                rgb, pose = line.strip().split(' ')
                self.rgb_paths.append(rgb)
                self.pose_paths.append(pose)
        self.indexs = np.arange(len(self.rgb_paths))
        self.total = len(self.rgb_paths) * self.opt.training_images * self.opt.training_hyps
        self._new_epoch()

    def _getCoordImg(rgb, sampling):
    	start_time = time.time()
    	patches = []
    	for i in xrange(sampling):
    		origX = sampling[i][0]
    		origY = sampling[i][1]

    		minx = origX - self.opt.input_size/2
    		maxx = origX + self.opt.input_size/2
    		miny = origY - self.opt.input_size/2
    		maxy = origY + self.opt.input_size/2

    		patches.append(rgb[miny:maxy,minx:maxx,:])

    	prediction = self.ol.predict(self.sess, patches)
    	prediction *= 1000 # conversion of meters to millimeters
    	prediction = prediction.reshape((self.obj_size,self.obj_size,3))

        print 'CNN prediction took {}ms'.format(time.time()-start_time*1000)
        return prediction

	def _new_epoch(self):

        def _getRandHyp(guassRot, gaussTrans):
            trans = np.random.normal(0, gaussTrans, size=3)
            rotAxis = np.random.normal(0, 1, size=3)
            rotAxis = rotAxis / np.linalg.norm(rotAxis)
            rotAxis = rotAxis * np.random.normal(0, guassRot) * math.pi / 180.0
            rot,_ = cv2.Rodrigues(rotAxis)
            transform = np.eye(4)
            transform[:3,:3] = rot
            transform[:3,3] = trans
            return transform

        def _createScore(poseGT):
            driftLevel = np.random.randint(2)
            if driftLevel == 0:
                poseNoise = np.matmul(poseGT, self._getRandHyp(2, 2))
            else:
                poseNoise = np.matmul(poseGT, self._getRandHyp(10, 100))
            poseNoise_cv = our2cv(poseNoise)
            diffMap = getDiffMap(poseNoise, estObj, sampling, properties.getCamMat())
            angularDistance, translationDistance = calcDistance(poseGT, poseNoise)
            correct = 0
            if angularDistance<5 and translationDistance<50:
                correct = 1
            score = -self.opt.obj_temperature * np.max(angularDistance, translationDistance/10)
            return diffMap, score, correct

		start_time = time.time()
		np.random.shuffle(self.indexs)
		self.data = []
		self.scores = []
        total_correct = 0.0
		for i in xrange(self.opt.training_images):
			curi = self.indexs[i]
			rgb = cv2.imread(path.join(self.opt.dataset_dir, self.rgb_paths[curi]))
			pose = getInfo(path.join(self.opt.dataset_dir, self.pose_paths[curi]))
			sampling = stochasticSubSample(self.img_height, self.img_width, self.opt.obj_size, self.input_size)
			estObj = self._getCoordImg(rgb, sampling)
            pool = Pool(16)
            res = pool.map(_createScore, np.repeat(pose,self.opt.training_hyps))
            pool.close()
            pool.join()
            diffMap, score, correct = zip(*res)
            self.data+=diffMap
            self.scores+=score
            total_correct += np.sum(correct)

        self.step = 0
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
