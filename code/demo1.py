from __future__ import division
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import cv2
import glob
import uuid
import math
import time
import copy
import NumericLayers
import layers

np.set_printoptions(threshold=np.inf)

CNN_OBJ_MAXINPUT = 100.0
EPS = 0.00000001
REFTIMES = 8
INLIERTHRESHOLD2D = 10
INLIERCOUNT = 20
SKIP = 100
HYPNUM = 256
PI = 3.1415926
BRG_SIZE = 42
OBJ_SIZE = 40
SAMPLESIZE = OBJ_SIZE * OBJ_SIZE

rotGT_np = np.array([[1.0, 10.0, 3.0], [5.5, 2.5, 10.0], [8.2, 3.4, 11.0]], dtype=np.float64)
tranGT_np = np.array([[5.5, 8.8, 4.4]], dtype=np.float64)
rotGT = tf.constant(rotGT_np)
tranGT = tf.constant(tranGT_np)

D = np.array([[0, 0, 0, 0, 0]], np.float64)
camera_matrx = [[1, 0, 1], [0, 1, 1], [0, 0, 1]]
np_cmat = np.array(camera_matrx, np.float64)
cmat = tf.constant(np_cmat)
distcoeff = tf.constant(D)
shuffleIdx1 = np.zeros((REFTIMES, SAMPLESIZE), dtype=np.int)
for i in range(shuffleIdx1.shape[0]):
	shuffleIdx1[i] = np.arange(SAMPLESIZE)
	np.random.shuffle(shuffleIdx1[i])
shuffleIdx1 = np.cast['int32'](shuffleIdx1)
shuffleIdx = tf.constant(shuffleIdx1)

inputImg = np.random.randint(0, 255, size=(480, 640, 3))

e1 = time.time()
sampling2D_np = layers.stochasticSubSample(inputImg, 40, 42)

'''

DepthNet

'''
sampling3D_np = np.random.uniform(0, 20, size=(OBJ_SIZE*OBJ_SIZE, 3))
sampling2D = tf.constant(sampling2D_np)
sampling3D = tf.constant(sampling3D_np)
objPts, imgPts, objIdx = layers.Get4PointLayer([sampling3D, sampling2D, cmat, distcoeff])
hyps = layers.PnPLayer([objPts, imgPts, cmat, distcoeff])
diffMaps = layers.ReprojectionLayer([sampling3D, sampling2D, hyps, cmat, distcoeff])
reflayer = layers.RefineLayer(REFTIMES, HYPNUM, SAMPLESIZE)
refhyps = reflayer.refine([sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeff])
rot_list = []
tran_list = []
err_list = []
for i in range(HYPNUM):
	rot, tran, jac = layers.ConvertFormatLayer([tf.expand_dims(refhyps[i], axis=0)])
	rot_list.append(rot)
	tran_list.append(tran)
for i in range(HYPNUM):
	err = layers.ExpectedMaxLoss([rotGT, tranGT, rot_list[i], tran_list[i]])
	err_list.append(err)

with tf.Session() as sess:
	print sess.run(err_list[0])