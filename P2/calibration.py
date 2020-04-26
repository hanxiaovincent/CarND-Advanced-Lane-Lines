import numpy as np
import cv2

def calibrate(images, patternSize, objectPoints):
	"""
	Calibrate the camera using checkerboard calibration.

	Parameters:
	images: Input images.
	patternSize: The size of checkerboard pattern.
	objectPoints: The object points in a 3D real world coordinate.

	Returns:
	cameraMatrix: Camera matrix.
	distCoeffs: Distortion correction coefficient.
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
	Undistort image.

	Parameters:
	image: Input image.
	cameraMatrix: Cam matrix.
	distCoeffs: Distortion coefficients.

	Returns:
	Undistorted image.
	"""
	return cv2.undistort(image, cameraMatrix, distCoeffs, None, None)
	