# USAGE
# python detect_mrz.py --images examples

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import math
import os
import pytesseract
from PIL import Image
import utils

def sortBox(box):
	def sorter(x): return (x[0], x[1])
	newBox = sorted(box.copy(), key=sorter)

	if newBox[0][1] < newBox[1][1]:
		newBox[0][1], newBox[1][1] = newBox[1][1], newBox[0][1]

	if newBox[2][1] < newBox[3][1]:
		newBox[2][1], newBox[3][1] = newBox[3][1], newBox[2][1]

	return newBox

def findDis(v1, v2):
	return math.sqrt((v1[0] - v2[0]) * (v1[0] - v2[0]) + (v1[1] - v2[1]) * (v1[1] - v2[1]))


def subimage(image, minRect):
	''' 
	Rotates OpenCV image around center with angle theta (in deg)
	then crops the image according to width and height.
	'''
	# Uncomment for theta in radians
	#theta *= 180/np.pi

	center = minRect[0]
	width = min(int(minRect[1][0]), image.shape[1])
	height = min(int(minRect[1][1]), image.shape[0])
	theta = minRect[2]

	if theta < -45:
		theta += 90

	# cv2.warpAffine expects shape in (length, height)
	shape = (image.shape[1], image.shape[0])

	matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
	image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

	x = max(int(center[0] - width/2), 0)
	y = max(int(center[1] - height/2), 0)

	image = image[y:y+height, x:x+width]

	return image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,help="path to images directory")
args = vars(ap.parse_args())

# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

counter = 1

# loop over the input image paths
for imagePath in paths.list_images(args["images"]):
	# load the image, resize it, and convert it to grayscale

	path, dirs, files = next(os.walk(args["images"] + "/"))
	totalFiles = len(files)
	print('==========================================================')
	print('counter = ', counter, '/', totalFiles)
	print('image: ', imagePath)
	counter += 1

	image = cv2.imread(imagePath)
	image = imutils.resize(image, height=600)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# smooth the image using a 3x3 Gaussian, then apply the blackhat
	# morphological operator to find dark regions on a light background
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

	# compute the Scharr gradient of the blackhat image and scale the
	# result into the range [0, 255]
	gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

	# apply a closing operation using the rectangular kernel to close
	# gaps in between letters -- then apply Otsu's thresholding method
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	thresh = cv2.threshold(
		gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# perform another closing operation, this time using the square
	# kernel to close gaps between lines of the MRZ, then perform a
	# serieso of erosions to break apart connected components
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
	thresh = cv2.erode(thresh, None, iterations=4)

	# during thresholding, it's possible that border pixels were
	# included in the thresholding, so let's set 5% of the left and
	# right borders to zero
	p = int(thresh.shape[1] * 0.01)
	thresh[:, 0:p] = 0
	thresh[:, thresh.shape[1] - p:] = 0

	# print("image.shape = ", image.shape)
	# print("p = ", p)

	# cv2.imshow("Test", thresh)

	# denoise - clear all connections from mrz to others horizontal
	for i in range(thresh.shape[0]):  # traverses through height of the image
		countWhite = 0
		startIndex = 0
		# traverses through width of the image
		for j in range(thresh.shape[1]):
			if thresh[i][j] == 255:  # 255 -> white
				countWhite += 1
			else:
				if countWhite > 0 and countWhite < 80:
					for k in range(startIndex, j + 1):
						thresh[i][k] = 0
				startIndex = j
				countWhite = 0

	# connect nearby contours
	denoiseKernel = np.ones((3, 3), np.uint8)
	thresh = cv2.dilate(thresh, denoiseKernel, iterations=4)

	# cv2.imshow("Denoise", thresh)

	# find contours in the thresholded image and sort them by their size
	cnts = cv2.findContours(
		thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[2], reverse=True)

	newCnts = []
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		if w / h > 6.0:
			newCnts.append(c)
	cnts = newCnts

	print('len(cnts) = ', len(cnts))

	# for c in cnts:
	# 	testRect = cv2.minAreaRect(c)
	# 	testBox = cv2.boxPoints(testRect)
	# 	testBox = np.int0(testBox)
	# 	cv2.drawContours(image, [testBox], 0, (255, 0, 0), 1)

	if len(cnts) > 0:

		rect0 = cv2.minAreaRect(cnts[0])
		box0 = cv2.boxPoints(rect0)
		box0 = np.int0(box0)
		box0 = sortBox(box0)
		w0 = rect0[1][0]
		h0 = rect0[1][1]
		angle = rect0[2]
		print('rect0 = ', rect0)
		print('box0 = ', box0)
		print('w0 = ', w0)

		if len(cnts) > 1:
			rect1 = cv2.minAreaRect(cnts[1])
			box1 = cv2.boxPoints(rect1)
			box1 = np.int0(box1)
			box1 = sortBox(box1)
			w1 = rect1[1][0]
			h1 = rect1[1][1]
			print('box1 = ', box1)

			print('abs(w1 - w0) / w0 = ', abs(w1 - w0) / w0)
			print('abs(h1 - h0) / h0 = ', abs(h1 - h0) / h0)

			if abs(w1 - w0) / w0 < 0.1 and abs(h1 - h0) / h0 < 0.2:
				box = None
				print('w0 = ', w0, ' && w1 = ', w1)
				if box0[1][1] < box1[1][1]:
					box = np.array([
						box0[1],
						box0[3],
						box1[2],
						box1[0]
					])
				else:
					box = np.array([
						box1[1],
						box1[3],
						box0[2],
						box0[0]
					])
				print('new box = ', box)
			else:
				box = np.array([
					box0[0],
					box0[2],
					box0[3],
					box0[1]
				])
				box = np.int0(box)
		else:
			print('draw box0 = ', box0)
			box = np.array([
				box0[0],
				box0[2],
				box0[3],
				box0[1]
			])

		outerBox = sortBox(box.copy())

		w = findDis(outerBox[0], outerBox[2])
		h = findDis(outerBox[0], outerBox[1])

		# padding box
		pX = int(w * 0.03)
		pY = int(h * 0.1)

		# bottom left
		outerBox[0][0] -= pX
		outerBox[0][1] += pX
		# top left
		outerBox[1][0] -= pX
		outerBox[1][1] -= pX
		# bottom right
		outerBox[2][0] += pX
		outerBox[2][1] += pX
		# top right
		outerBox[3][0] += pX
		outerBox[3][1] -= pX

		box = np.array([
			outerBox[0],
			outerBox[1],
			outerBox[3],
			outerBox[2]
		])
		
		center = ((outerBox[0][0] + outerBox[3][0]) / 2,
				  (outerBox[0][1] + outerBox[3][1]) / 2)
		
		minRect = (center, (w + pX * 2, h + pX * 2), angle)
		print('image size = ', (image.shape[1], image.shape[0]))
		print('minRect = ', minRect)
		print('box = ', outerBox)
		print('outerBox = ', outerBox)
		roi = subimage(image, minRect)
		cv2.drawContours(image, [box], 0, (0, 255, 0), 1)

		# mrz = recognizeText(roi)
		# print(mrz)

		# show the output images
		# cv2.imshow("Image", image)
		# cv2.imshow("ROI", roi)
		# print('recognized roi = ', utils.recognizeText(roi))
		# cv2.imwrite('./rois/roi' + str(counter) +'.jpg', roi)
		# print('recognized roi = ', utils.recognizeText(roi))

		# sharpened_image = utils.unsharp_mask(roi)
		# cv2.imwrite('sharpened-image.jpg', sharpened_image)
		# print('recognized sharpened = ', utils.recognizeText(sharpened_image))

		# unshadow_image = utils.remove_shadow(roi); 
		# cv2.imwrite('unshadow-image.jpg', unshadow_image)
		# print('recognized shadow = ', utils.recognizeText(unshadow_image))

		# meanc = utils.mean_c(roi)
		# cv2.imwrite('meanc-image.jpg', meanc)
		# print('recognized meanc = ', utils.recognizeText(meanc))

		# remove_noise = utils.remove_noise(roi)
		# cv2.imwrite('remove-noise-image.jpg', remove_noise)
		# print('remove noise = ', utils.recognizeText(remove_noise))

		# cv2.imshow('remove_noise', remove_noise)

		cv2.waitKey(0)
		cv2.destroyAllWindows()
