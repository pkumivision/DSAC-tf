import re
import numpy as np
import math
import cv2
import time

from config import cfg

def getInfo(pose_txt):
    a = []
    with open(pose_txt) as f:
        pattern = re.compile('\s+')
        for line in f.readlines():
            ss=re.split(pattern, line.strip())
            for s in ss:
                a.append(float(s))

    a = np.array(a).reshape(4,4)
    ts = [0.6880049706, 0.333539999278, 2.23485151692]
    for i in xrange(3):
        a[i][3] -= ts[i]

    correction = np.array([[1,0,0,0],
                            [0,-1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]])
    a = np.matmul(a,correction)
    for i in xrange(3):
        a[i][3] *= 1000
    return np.linalg.inv(a)

def stochasticSubSample(rgb, targetSize, patchSize):
    h, w, _ = rgb.shape
    xStride = (w - patchSize) / float(targetSize)
    yStride = (h - patchSize) / float(targetSize)
    x, y = np.meshgrid(np.arange(targetSize) * xStride + patchSize / 2,np.arange(targetSize) * yStride + patchSize / 2)
    x = x.reshape(-1)
    y = y.reshape(-1)
    x = np.random.rand(targetSize * targetSize) * xStride + x
    y = np.random.rand(targetSize * targetSize) * yStride + y
    sampling = np.array(zip(x,y)).astype(np.int)

    patches = []
    for i in xrange(len(sampling)):
        origX = sampling[i][0]
        origY = sampling[i][1]

        minx = origX - patchSize/2
        maxx = origX + patchSize/2
        miny = origY - patchSize/2
        maxy = origY + patchSize/2

        patches.append(rgb[miny:maxy,minx:maxx,:])
    return sampling, patches

def our2cv(trans):
    tmp = trans.copy()
    rmat = tmp[:3,:3]
    rmat[1:3] = -rmat[1:3]
    rvec, _ = cv2.Rodrigues(rmat)
    rvec = np.squeeze(rvec)

    tvec = tmp[:3,3]
    tvec[1:3] = -tvec[1:3]
    return np.concatenate([rvec, tvec])

# @param hyp, cvForms
def getDiffMap(hyp, sampling3D, sampling2D, cmat, distcoeffs=None):
    points2D = sampling2D
    points3D = sampling3D
    projections, _ = cv2.projectPoints(sampling3D, hyp[0:3], hyp[3:6], cmat, distcoeffs)
    (m, _, n) = projections.shape
    projections = projections.reshape(m, n)
    diffPt = points2D - projections
    diffMap = np.minimum(np.linalg.norm(diffPt, axis = 1, keepdims = False), cfg.CNN_OBJ_MAXINPUT)
    return diffMap

def calcDistance(pose1, pose2):
    rot1 = pose1[:3,:3]
    rot2 = pose2[:3,:3]
    rotDiff = np.matmul(rot1, np.linalg.inv(rot2))
    trace = np.trace(rotDiff)
    trace = np.min([3.0, np.max([-1.0,trace])])
    angularDistance = 180.0*np.arccos((trace-1.0)/2.0)/math.pi

    trans1 = pose1[:,3]
    trans2 = pose2[:,3]
    translationDistance = np.linalg.norm(trans1-trans2)
    return angularDistance, translationDistance

def getCoordImg(patches, sess, objLearner, opt):
    start_time = time.time()
    prediction = objLearner.predict(sess, patches)
    prediction *= 1000 # conversion of meters to millimeters
    prediction = prediction.reshape((opt.obj_size, opt.obj_size, 3))

    if opt.time_info:
        print 'CNN prediction took {}ms'.format((time.time()-start_time)*1000)
    return prediction
