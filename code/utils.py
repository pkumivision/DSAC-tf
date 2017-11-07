import re
import numpy as np
import math

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

def stochasticSubSample(h, w, targetSize, patchSize):
    rows = h
    cols = w
    sampling = np.zeros((targetSize, targetSize, 2))
    xStride = (cols - patchSize) / targetSize
    yStride = (rows - patchSize) / targetSize
    sampleX = 0
    minX = patchSize / 2
    x = xStride + patchSize / 2
    while x <= cols - patchSize / 2 + 1:
        sampleY = 0
        y = yStride + patchSize / 2
        minY = patchSize / 2
        while y <= rows - patchSize / 2 + 1:
            curX = np.random.randint(minX, x)
            curY = np.random.randint(minY, y)
            sampling[sampleY][sampleX][0] = curX
            sampling[sampleY][sampleX][1] = curY
            sampleY += 1
            minY = y
            y += yStride
        sampleX += 1
        minX = x
        x += xStride
    return sampling.reshape(-1, 2)

def our2cv(trans):
    rmat = trans[:3,:3]
    rmat[1:3] = -rmat[1:3]
    rvec, _ = cv2.Rodrigues(rmat)

    tvec = trans[:3,3]
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
    angularDistance = 180.0*np.arcos((trace-1.0)/2.0)/math.pi

    trans1 = pose1[:,3]
    trans2 = pose2[:,3]
    translationDistance = np.linalg.norm(trans1-trans2)
    return angularDistance, translationDistance