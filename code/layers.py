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

np.set_printoptions(threshold=np.inf)

CNN_OBJ_MAXINPUT = 100.0
MAXLOSS = 10000000.0
EPS = 0.00000001
REFTIMES = 8
INLIERTHRESHOLD2D = 10
INLIERCOUNT = 20
SKIP = 100
HYPNUM = 10
SAMPLESIZE = 1000
PI = 3.1415926

def containNan(obj):
	for i in obj:
		if math.isnan(i):
			return True
	return False

def stochasticSubSample(inputImg, targetSize, patchSize):
	(rows, cols, channels) = inputImg.shape
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

def cv2our(cvTrans):
	rots = cvTrans[0:3]
	trans = cvTrans[3:6]
	rmat, _ = cv2.Rodrigues(rots)
	tpt = copy.deepcopy(trans)
	rmat[1][:] = -rmat[1][:]
	rmat[2][:] = -rmat[2][:]
	tpt[1] = -tpt[1]
	tpt[2] = -tpt[2]
	
	if cv2.determinant(rmat) < 0:
		tpt = -tpt
		rmat = -rmat
	
	for i in tpt:
		if math.isnan(i):
			tpt = np.zeros((3,))

	tpt = tpt.reshape(3,1)
	return np.append(rmat, tpt, axis = 1)

def getRodVecAndTrans(rots, trans):
	rv, _ = cv2.Rodrigues(rots)
	res = np.zeros((6,), np.float32)
	res[0] = rv[0]
	res[1] = rv[1]
	res[2] = rv[2]
	res[3] = trans[0]
	res[4] = trans[1]
	res[5] = trans[2]
	return res

def getRots(hyp):
	return hyp[0:3, 0:3]

def getTrans(hyp):
	return hyp[0:3, 3]

def getDiffMap(hyp, sampling3D, sampling2D, cmat, distcoeffs):
	'''
	getDiffMap

	@brief  use the estimated hyps to reproject 3D points into 2D, and calculate the difference between the project2D and groudtruth2D

	'''
	points2D = sampling2D
	points3D = sampling3D
	projections, _ = cv2.projectPoints(sampling3D, hyp[0:3], hyp[3:6], cmat, distcoeffs)
	(m, _, n) = projections.shape
	projections = projections.reshape(m, n)
	diffPt = points2D - projections
	diffMap = np.minimum(np.linalg.norm(diffPt, axis = 1, keepdims = False), CNN_OBJ_MAXINPUT)
	return diffMap

def Get4PointLayer(inputs):
	'''
	Get4Point Layer

	@brief  pick a batch of points from sampled points
	@param  sampling3D, (n,3) sampled 3D points
	@param  sampling2D, (n,2) sampled 2D points
	@param  cmat,       (3,3) camera instrincs
	@output objPts,     (h,4,3) picked 3D points
	@output imgPts,     (h,4,2) picked 2D points
	@output objIdx,     (h,4)   the index of picked points in the origin sampling pool, which will be used later
	'''
	def _get4point(sampling3D, sampling2D, cmat, distcoeffs):
		objPts = np.zeros((HYPNUM, 4, 3), np.float64)
		imgPts = np.zeros((HYPNUM, 4, 2), np.float64)
		objIdx = np.zeros((HYPNUM, 4), np.int32)
		e1 = time.time()
		for h in range(HYPNUM):
			#print(h)
			alreadyChosen = np.zeros((sampling3D.shape[0]), np.int)
			while True:
				propObj = np.zeros((4, 3))
				propImg = np.zeros((4, 2))
				propIdx = np.zeros((4,))
				for i in range(4):
					idx = np.random.randint(sampling3D.shape[0])
					#print(idx)
					if alreadyChosen[idx] == 1:
						continue
					propObj[i] = sampling3D[i]
					propImg[i] = sampling2D[i]
					propIdx[i] = idx
					alreadyChosen[idx] = 1

				done, rot, tran = cv2.solvePnP(propObj, propImg, cmat, distcoeffs)
				hyp = np.append(rot, tran)

				projections, _ = cv2.projectPoints(propObj, hyp[0:3], hyp[3:6], cmat, distcoeffs)
				(m, _, n) = projections.shape
				projections = projections.reshape(m, n)
				found = True
				for i in range(4):
					if np.linalg.norm(projections[i] - propImg[i]) < INLIERTHRESHOLD2D:
						continue
					found = False
					break
				if found:
					objPts[h] = propObj
					imgPts[h] = propImg
					objIdx[h] = propIdx
					break
		print "get 4 point use", time.time() - e1, "s"
		return objPts, imgPts, objIdx

	def _get4point_grad(sampling3D, objIdx, grad):
		dSample = np.zeros_like(sampling3D)
		for h in range(objIdx.shape[0]):
			dSample[objIdx[h]] = grad[h]
		return dSample

	def _get4point_grad_op(op, grad1, grad2, grad3):
		objIdx = op.outputs[2]
		sampling3D = op.inputs[0]
		tf_dSample = tf.py_func(_get4point_grad, [sampling3D, objIdx, grad1], tf.float64)
		return [tf_dSample, None, None, None]

	grad_name = "Get4pointGrad_" + str(uuid.uuid4())
	tf.RegisterGradient(grad_name)(_get4point_grad_op)
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": grad_name}):
		output1, output2, output3 = tf.py_func(_get4point, inputs, [tf.float64, tf.float64, tf.int32])
	return output1, output2, output3

def PnPLayer(inputs):
	'''
	PnPlayer

	@param  objPts (h,4,3) picked points, generated by DepthCNN
	@param  imgPts (h,4,2) picked points
	@cmat   camera instrincs
	@distcoeffs

	@output hyps   (h, 6)
	'''

	def _pnp(objPts, imgPts, cmat, distcoeffs):
		e1 = time.time()
		res = np.zeros((objPts.shape[0], 6),np.float64)
		NumericLayers.solvePnP(objPts, imgPts, cmat, res, 2)
		return res
	
	def _pnp_grad(objPts, imgPts, cmat, distcoeffs, grad):
		(h_len, m, n) = objPts.shape
		jacobean = np.zeros((h_len, m, n), np.float64)
		eps = 1
		obj_0 = np.tile(objPts, (12, 1, 1))
		obj_1 = copy.deepcopy(obj_0)

		for i in range(4):
			for j in range(3):
				obj_0[(i*3+j)*h_len:(i*3+j+1)*h_len, i, j] -= eps
				obj_1[(i*3+j)*h_len:(i*3+j+1)*h_len, i, j] += eps
		
		tmp_obj = np.concatenate((obj_0, obj_1))
		tmp_img = np.tile(imgPts, (24, 1, 1))
		tmp_hyp = np.zeros((tmp_obj.shape[0], 6), np.float64)

		NumericLayers.solvePnP(tmp_obj, tmp_img, cmat, tmp_hyp, 2)
		
		res_0 = tmp_hyp[0:h_len*12, :]
		res_1 = tmp_hyp[h_len*12:h_len*24,:]

		for i in range(4):
			for j in range(3):
				jacobean[:, i, j] = np.sum(grad * (res_1[(i*3+j)*h_len:(i*3+j+1)*h_len,:] - res_0[(i*3+j)*h_len:(i*3+j+1)*h_len,:]) / (2 * eps), axis = 1)
		
		return jacobean

	def _pnp_grad_op(op, grad):
		aa = op.inputs[0]
		bb = op.inputs[1]
		cc = op.inputs[2]
		dd = op.inputs[3]
		p_grad = tf.py_func(_pnp_grad, [aa, bb, cc, dd, grad], tf.float64)
		return [p_grad, None, None, None]

	grad_name = "PnPGrad_" + str(uuid.uuid4())
	tf.RegisterGradient(grad_name)(_pnp_grad_op)
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": grad_name}):
		output = tf.py_func(_pnp, inputs, tf.float64)
	return output

def ReprojectionLayer(inputs):
	'''
	ReprojectionLayer

	@brief get diffmap through reprojection
	@param sampling3D, (n,3) 3D coordinates
	@param sampling2D, (n,2) 2D coordinates
	@param hyps,       (h,6) containing rvec and tvec
	@param cmat,       (3,3) camera instrincs
	@param distcoeffs
	@output diffMaps   (h,n)
	'''

	def _reprojection(sampling3D, sampling2D, hyps, cmat, distcoeffs):
		#print(objPts)
		(h, _) = hyps.shape
		(n, _) = imgPts.shape
		diffMaps = np.zeros((h, n))
		for i in range(h):
			diffMaps[i] = getDiffMap(hyps[i], sampling3D, sampling2D, cmat, distcoeffs)
		return diffMaps

	# @param hyp, jpForms, 3x3 + 3x1, rmat and tvec
	def _dProject_wrt_obj(sampling3D, sampling2D, hyp, cmat, grad):
		# camera instrincs
		f = cmat[0][0]
		ppx = cmat[0][2]
		ppy = cmat[1][2]
		rot = getRots(hyp)
		tran = getTrans(hyp)
		jacobean = np.zeros_like(sampling3D)

		for i in range(sampling3D.shape[0]):
			objMat = np.dot(rot, sampling3D[i]) + tran

			if np.abs(objMat[2]) < EPS:
				continue
			px = -f * objMat[0] / objMat[2] + ppx
			py = f * objMat[1] / objMat[2] + ppy

			err = np.sqrt((sampling2D[i][0] - px) ** 2 + (sampling2D[i][1] - py) ** 2)
			if err > CNN_OBJ_MAXINPUT:
				continue

			err += EPS

			pxdx = -f * rot[0][0] / objMat[2] + f * objMat[0] / objMat[2] / objMat[2] * rot[2][0]
			pydx = f * rot[1][0] / objMat[2] - f * objMat[1] / objMat[2] / objMat[2] * rot[2][0]
			dx = 0.5 / err * (2 * (sampling2D[i][0] - px) * (-pxdx) + 2 * (sampling2D[i][1] - py) * (-pydx))

			pxdy = -f * rot[0][1] / objMat[2] + f *objMat[0] / objMat[2]/ objMat[2] * rot[2][1]
			pydy = f * rot[1][1] / objMat[2] - f * objMat[1] / objMat[2] / objMat[2] * rot[2][1]
			dy = 0.5 / err * (2 * (sampling2D[i][0] - px) * (-pxdy) + 2 * (sampling2D[i][1] - py) * (-pydy))

			pxdz = -f * rot[0][2] / objMat[2] + f *objMat[0] / objMat[2]/ objMat[2] * rot[2][2]
			pydz = f * rot[1][2] / objMat[2] - f * objMat[1] / objMat[2] / objMat[2] * rot[2][2]
			dz = 0.5 / err * (2 * (sampling2D[i][0] - px) * (-pxdz) + 2 * (sampling2D[i][1] - py) * (-pydz))

			jacobean[i][0] = dx * grad[i]
			jacobean[i][1] = dy * grad[i]
			jacobean[i][2] = dz * grad[i]

		return jacobean

	# @param hyp, jpForms, 3x3+3x1, rmat and tvec
	def _dProject_wrt_hyp(sampling3D, sampling2D, hyp, cmat, grad):
		f = cmat[0][0]
		ppx = cmat[0][2]
		ppy = cmat[1][2]
		rot = getRots(hyp)
		tran = getTrans(hyp)
		jacobean = np.zeros((6,))
		dRdH = np.zeros((9, 3))
		rot1 = copy.deepcopy(rot)
		rot1[1][:] = -rot1[1][:]
		rot1[2][:] = -rot1[2][:]
		rod, _ = cv2.Rodrigues(rot1)
		rot1, dRdH = cv2.Rodrigues(rod)
		dRdH = dRdH.T

		for i in range(sampling3D.shape[0]):
			objMat = sampling3D[i]
			eyeMat = np.dot(rot, sampling3D[i]) + tran
			if np.abs(eyeMat[2]) < EPS:
				print("fire")
				continue
			
			px = -f * eyeMat[0] / eyeMat[2] + ppx
			py = f * eyeMat[1]/ eyeMat[2] + ppy

			err = np.sqrt((sampling2D[i][0] - px)**2 + (sampling2D[i][1] - py)**2)

			if err > CNN_OBJ_MAXINPUT:
				continue

			err += EPS

			# derivative of error wrt projection
			dNdP = np.zeros((2,))
			dNdP[0] = -1 / err * (sampling2D[i][0] - px) * grad[i]
			dNdP[1] = -1 / err * (sampling2D[i][1] - py) * grad[i]
			# derivative of projection wrt rotation matrix
			dPdR = np.zeros((2, 9))
			
			dPdR[0][0:3] = -f * objMat / eyeMat[2]
			dPdR[1][3:6] = f * objMat / eyeMat[2]
			dPdR[0][6:9] = f * eyeMat[0] / eyeMat[2] / eyeMat[2] * objMat
			dPdR[1][6:9] = -f * eyeMat[1] / eyeMat[2]/ eyeMat[2] * objMat
			dPdR[:,3:9] = -dPdR[:, 3:9]
			
			# derivative of rotation matrix wet rodriguez vector
			# combine, derivative of error wrt Rodriguez vector
			dNdH = np.dot(np.dot(dNdP, dPdR), dRdH)

			# derivative of projection wet translation vector
			dPdT = np.zeros((2,3))
			dPdT[0][0] = -f / eyeMat[2]
			dPdT[1][1] = f / eyeMat[2]
			dPdT[0][2] = f * eyeMat[0] / eyeMat[2] / eyeMat[2]
			dPdT[1][2] = -f * eyeMat[1] / eyeMat[2] / eyeMat[2]

			dNdT = np.dot(dNdP, dPdT)
			dNdT[1:3] = -dNdT[1:3]
			jacobean[0:3] += dNdH
			jacobean[3:6] += dNdT

		return jacobean

	def _reprojection_grad(sampling3D, sampling2D, hyps, cmat, distcoeffs, grad):
		dObj = np.zeros_like(sampling3D)
		dHyp = np.zeros_like(hyps)
		for i in range(hyps.shape[0]):
			ourHyp = cv2our(hyps[i])
			dObj += _dProject_wrt_obj(sampling3D, sampling2D, ourHyp, cmat, grad[i])
		for i in range(hyps.shape[0]):
			ourHyp = cv2our(hyps[i])
			dHyp[i] = _dProject_wrt_hyp(sampling3D, sampling2D,  ourHyp, cmat, grad[i])
		return dObj, dHyp

	def _reprojection_grad_op(op, grad):
		sampling3D = op.inputs[0]
		sampling2D = op.inputs[1]
		hyps = op.inputs[2]
		cmat = op.inputs[3]
		distcoeffs = op.inputs[4]
		objGrad, hypGrad = tf.py_func(_reprojection_grad, [sampling3D, sampling2D, hyps, cmat, distcoeffs, grad], [tf.float64, tf.float64])
		return [objGrad, None, hypGrad, None, None]

	grad_name = "reprojectionGrad_" + str(uuid.uuid4())
	tf.RegisterGradient(grad_name)(_reprojection_grad_op)
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": grad_name}):
		output = tf.py_func(_reprojection, inputs, tf.float64)
	return output

def ConvertFormatLayer(inputs):
	'''
	ConvertFormatLayer

	@brief  change hyp format of 1x6 to 3x3(rot matrix) and 3x1(tran vec)
	@param  one hyp : (1,6)
	@output rmat:     (3,3)
	@output tvec:     (1,3)

	'''

	def _convert(hyp):
		rot = hyp[0][0:3]
		tran = hyp[0][3:6]
		rmat, jac = cv2.Rodrigues(rot)
		tran = np.expand_dims(tran, axis=0)
		return rmat, tran, jac

	def _convert_grad_op(op, grad1, grad2, grad3): # grad1 (3,3)  grad2 (1,3)
		hyp = op.inputs[0]
		jac = op.outputs[2]
		rot = hyp[0][0:3]
		tran = hyp[0][3:6]
		drot = tf.matmul(jac, tf.reshape(grad1,(9,1)))
		drot = tf.transpose(drot)
		D = tf.concat((drot, grad2), axis=1)

		return [D]

	grad_name = "ConvertFormatLayer_" + str(uuid.uuid4())
	tf.RegisterGradient(grad_name)(_convert_grad_op)
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": grad_name}):
		rmat, tran, jac = tf.py_func(_convert, inputs, [tf.float64, tf.float64, tf.float64])
	return rmat, tran, jac

def ExpectedMaxLoss(inputs):
	'''
	ExpectedMaxLoss layer

	@brief, get final loss
	@param  rotGT: (3,3) rot matrix
	@param  tranGT: (1,3) tran vector
	@param  rotEst: (3,3) rot matrix
	@param  tranEst: (1,3) tran vector
	'''

	'''
	prepare for inverse operation, convert pose info into this format:
	[rot11, rot12, rot13, tran1]
	[rot21, rot22, rot23, tran2]
	[rot31, rot32, rot33, tran3]
	[0,     0,     0,     1   ]

	then can calculate inverse
	'''
	rotGT, tranGT, rotEst, tranEst = inputs
	appendix = tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float64)

	tranGT = tf.transpose(tranGT)
	poseGT = tf.concat([rotGT, tranGT], axis=1)
	poseGT = tf.concat([poseGT, appendix], axis=0)

	tranEst = tf.transpose(tranEst)
	poseEst = tf.concat([rotEst, tranEst], axis=1)
	poseEst = tf.concat([poseEst, appendix], axis=0)

	''' calculate inverse '''
	invGT = tf.matrix_inverse(poseGT)
	invRotGT = invGT[0:3,0:3]
	invTranGT = invGT[0:3, 3]

	invEst = tf.matrix_inverse(poseEst)
	invRotEst = invEst[0:3, 0:3]
	invTranEst = invEst[0:3, 3]

	''' rot error '''
	rotDiff = tf.matmul(invRotGT, invRotEst)
	trace = tf.trace(rotDiff)
	trace = tf.clip_by_value(trace, -1.0, 3.0)
	rotErr = 180 * tf.acos((trace - 1.0) / 2.0) / PI
	''' tran error '''
	tranErr = tf.norm(invTranGT - invTranEst)

	return tf.minimum(tf.maximum(rotErr, tranErr / 10), MAXLOSS)


class RefineLayer:
	def __init__(self, reftimes, hyp_num, sampling_num):
		self.reftimes = reftimes
		self.hyp_num = hyp_num
		self.sampling_num = sampling_num

	def refine(self, inputs):
		in_sampling3D, in_sampling2D, in_objPts, in_imgPts, in_hyp, in_objIdx, in_shuffleIdx, in_cmat, in_distcoeffs = inputs
		self.inlierMaps = np.zeros((self.hyp_num, self.sampling_num))
		self.inlierMaps = np.cast['int32'](self.inlierMaps)
		
		def _refine(sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeffs):
			refHyps = np.zeros(hyps.shape, dtype=np.float64)
			NumericLayers.refine(refHyps,
				sampling3D,
				sampling2D,
				objPts,
				imgPts,
				hyps,
				self.inlierMaps,
				objIdx,
				shuffleIdx,
				cmat,
				INLIERCOUNT)

			return refHyps

		def _refine_single(sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeffs):
			refHyps = np.zeros(hyps.shape, dtype=np.float64)
			for h in range(refHyps.shape[0]):
				diffmaps = getDiffMap(hyps[h], sampling3D, sampling2D, cmat, distcoeffs)
				for i in range(cfg.REFTIMES):
					inlier3D = []
					inlier2D = []
					for idx in shuffleIdx[i]:
						if diffmaps[idx] < cfg.INLIERTHRESHOLD2D:
							inlier3D.append(sampling3D[idx])
							inlier2D.append(sampling2D[idx])
						if len(inlier3D) > cfg.INLIERCOUNT:
							break
					if len(inlier3D) < 3:
						continue
					refineObj = np.array(inlier3D)
					refinePt = np.array(inlier2D)
					done, rot, tran = cv2.solvePnP(refineObj, refinePt, cmat, distcoeffs, False, cv2.SOLVEPNP_ITERATIVE if refineObj.shape[0] >= 4 else cv2.SOLVEPNP_P3P)
					if containNan(rot) or containNan(tran):
						break
					refHyps[h] = np.append(rot, tran)
					#print("i:", i, refHyps[h])
					diffmaps = getDiffMap(refHyps[h], sampling3D, sampling2D, cmat, distcoeffs)
			return refHyps

		# @param grad, dLoss wrt hyp, hx6, actually the same shape as hyps
		def _refine_grad(sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeffs, grad):
			# print(grad)
			# dRefine wrt the picked up points, which were used to generate pose before
			# numeric method
			eps = 1
			jacobean_obj = np.zeros_like(objPts, np.float64)
			jacobean_sample = np.zeros_like(sampling3D, np.float64)
			#inlierMaps = np.ones_like(self.inlierMaps)
			e1 = time.time()

			NumericLayers.dRefine(jacobean_obj,
				jacobean_sample,
				sampling3D,
				sampling2D,
				objPts,
				imgPts,
				self.inlierMaps,
				objIdx,
				shuffleIdx,
				cmat,
				grad,
				INLIERCOUNT,
				eps,
				SKIP)

			print time.time()-e1, "s"
			return jacobean_obj, jacobean_sample

		def _refine_grad_op(op, grad):
			[sampling3D, sampling2D, objPts, imgPts, thyp, objIdx, shuffleIdx, cmat, distcoeffs] = op.inputs

			dObj, dSample = tf.py_func(_refine_grad, [sampling3D, sampling2D, objPts, imgPts, thyp, objIdx, shuffleIdx, cmat, distcoeffs, grad], [tf.float64, tf.float64])		
			return [dSample, None, dObj, None, None, None, None, None, None]
		
		grad_name = "RefineGrad_" + str(uuid.uuid4())
		tf.RegisterGradient(grad_name)(_refine_grad_op)
		g = tf.get_default_graph()
		with g.gradient_override_map({"PyFunc": grad_name}):
			output = tf.py_func(_refine, inputs, tf.float64)
		return output




