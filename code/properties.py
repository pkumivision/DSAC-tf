import numpy as np

imageWidth = 640
imageHeight = 480

focalLength = 525.0
xShift = 0.0
yShift = 0.0

secondaryFocalLength = 585.0
rawXShift = 0.0
rawYShift = 0.0

def getSensorTrans():
    trans = np.array([[0.9998282978278875, 0.008186003849805841, 0.01662420535123559, -0.01393324413905143],
                      [-0.008090415588790156, 0.9999503986429169, -0.005809081637597863, 0.05228905046770047],
                      [-0.01667093393273896, 0.005673587475537798, 0.9998449331606215, 0.02712006871814571],
                      [0, 0, 0, 1]])
    return np.mat(trans)

def getCamMat():
	centerX = imageWidth/2 + xShift
	centerY = imageHeight/2 + yShift
	camMat = np.zeros((3,3), dtype=np.float)
	camMat[0,0] = focalLength
	camMat[1,1] = focalLength
	camMat[2,2] = 1.0

	camMat[0,2] = centerX
	camMat[1,2] = centerY
	return camMat
