import numpy as np
import cv2

def calibrate(images, patternSize, objectPoints):
	"""
	Calibrate the camera using checkerboard calibration.
	"""
	# extract corners
	objPoints = []
	imgPoints = []

	for img in images:
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, patternSize, None)

		if ret == True:
			objPoints.append(objectPoints)
			imgPoints.append(corners)

	# calibrate
	imgSize = images[0].shape[1::-1]
	retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
		objPoints, imgPoints, imgSize, None, None)

	# return only camera matrix and distCoeffs for this application
	return cameraMatrix, distCoeffs

def undistort(image, cameraMatrix, distCoeffs):
	"""
	Undistort image with camera matrix and distorion coefficients.
	"""
	return cv2.undistort(image, cameraMatrix, distCoeffs, None, None)
	