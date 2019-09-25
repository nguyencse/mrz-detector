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

def sortBox(box):
	sorter = lambda x: (x[0], x[1])
	newBox = sorted(box, key = sorter)

	if newBox[0][1] < newBox[1][1]:
		newBox[0][1], newBox[1][1] = newBox[1][1], newBox[0][1]

	if newBox[2][1] < newBox[3][1]:
		newBox[2][1], newBox[3][1] = newBox[3][1], newBox[2][1]

	return newBox

def cropRect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
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
	thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# perform another closing operation, this time using the square
	# kernel to close gaps between lines of the MRZ, then perform a
	# serieso of erosions to break apart connected components
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
	thresh = cv2.erode(thresh, None, iterations=4)

	# during thresholding, it's possible that border pixels were
	# included in the thresholding, so let's set 5% of the left and
	# right borders to zero
	p = int(image.shape[1] * 0.01)
	thresh[:, 0:p] = 0
	thresh[:, image.shape[1] - p:] = 0

	# print("image.shape = ", image.shape)
	# print("p = ", p)
 
	cv2.imshow("Test", thresh)

	# denoise
	for i in range(thresh.shape[0]): #traverses through height of the image
		countWhite = 0
		startIndex = 0
		for j in range(thresh.shape[1]): #traverses through width of the image
			if thresh[i][j] == 255: # 255 -> white
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

	cv2.imshow("Denoise", thresh)


	# find contours in the thresholded image and sort them by their size
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=lambda ctr : cv2.boundingRect(ctr)[2], reverse=True)

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
				cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
			else:
				box = np.array([
					box0[0],
					box0[2],
					box0[3],
					box0[1]
				])
				box = np.int0(box)
				cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
		else:
			print('draw box0 = ', box0)
			box = np.array([
				box0[0],
				box0[2],
				box0[3],
				box0[1]
			])
			cv2.drawContours(image, [box], 0, (0, 255, 0), 1)

	# get max width of contours
	# maxLen = -1
	# index = -1
	# for i in range(len(cnts)):
	# 	(x, y, w, h) = cv2.boundingRect(cnts[i])
	# 	if w > maxLen:
	# 		maxLen = w
	# 		index = i

	# # print('Max of index ', index)

	# if index != -1:
	# 	(x, y, w, h) = cv2.boundingRect(cnts[index])

	# 	ar = w / float(h)
	# 	crWidth = w / float(gray.shape[1])
	# 	# print('x = ', x, 'y = ', y, 'w = ', w, 'h = ', h)
	# 	# print('Shape of gray is ', gray.shape)

	# 	# check to see if the aspect ratio and coverage width are within
	# 	# acceptable criteria
	# 	# if ar > 5 and crWidth > 0.75:
	# 		# pad the bounding box since we applied erosions and now need
	# 		# to re-grow it
	# 	pX = int((x + w) * 0.03)
	# 	pY = int((y + h) * 0.03)
	# 	(x, y) = (x - pX, y - pY)
	# 	(w, h) = (w + (pX * 2), h + (pY * 2))

	# 	# extract the ROI from the image and draw a bounding box
	# 	# surrounding the MRZ
	# 	roi = image[y:y + h, x:x + w].copy()
	# 	# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# 	# print('ratio: width/height = ', w / h)

	# 	minRect = cv2.minAreaRect(cnts[index])

	# 	print("min rect = ", minRect)

	# 	box = cv2.boxPoints(minRect)
	# 	box = np.int0(box)
	# 	print("min box vertices = ", box)

	# 	# cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
	# 	if w / h > 6.0:
	# 		cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
	# 	else:
	# 		if minRect[1][0] < minRect[1][1]:
	# 			angle = minRect[2] - 180
	# 		else:
	# 			angle = minRect[2] + 90

	# 		angle =  angle * 3.14  / 180.0

	# 		print('angle', angle)

	# 		# sort vertices asccending
	# 		sorter = lambda x: (x[0], x[1])
	# 		sortedVertices = sorted(box, key = sorter)

	# 		if sortedVertices[0][1] < sortedVertices[1][1]:
	# 			sortedVertices[0][1], sortedVertices[1][1] = sortedVertices[1][1], sortedVertices[0][1]

	# 		if sortedVertices[2][1] < sortedVertices[3][1]:
	# 			sortedVertices[2][1], sortedVertices[3][1] = sortedVertices[3][1], sortedVertices[2][1]

	# 		print("min box vertices sorted = ", sortedVertices)

	# 		distance = minRect[1][0] / 13.5
	# 		newTopLeftX = sortedVertices[0][0] - math.cos(angle) * distance
	# 		newTopLeftY = sortedVertices[0][1] - math.sin(angle) * distance
	# 		newTopRightX = sortedVertices[2][0] - math.cos(angle) * distance
	# 		newTopRightY = sortedVertices[2][1] - math.sin(angle) * distance

	# 		newBox = np.array([
	# 			[newTopLeftX, newTopLeftY],
	# 			[sortedVertices[0][0], sortedVertices[0][1]],
	# 			[sortedVertices[2][0], sortedVertices[2][1]],
	# 			[newTopRightX, newTopRightY]
	# 		])
	# 		newBox = np.int0(newBox)
	# 		print('new box = ', newBox)
	# 		cv2.drawContours(image, [newBox], 0, (0, 0, 255), 1)
			
		# # loop over the contours
		# for c in cnts:
		# 	# compute the bounding box of the contour and use the contour to
		# 	# compute the aspect ratio and coverage ratio of the bounding box
		# 	# width to the width of the image
		# 	(x, y, w, h) = cv2.boundingRect(c)

		# 	ar = w / float(h)
		# 	crWidth = w / float(gray.shape[1])
		# 	print('x = ', x, 'y = ', y, 'w = ', w, 'h = ', h)
		# 	print('Shape of gray is ', gray.shape)

		# 	# check to see if the aspect ratio and coverage width are within
		# 	# acceptable criteria
		# 	# if ar > 5 and crWidth > 0.75:
		# 		# pad the bounding box since we applied erosions and now need
		# 		# to re-grow it
		# 	pX = int((x + w) * 0.03)
		# 	pY = int((y + h) * 0.03)
		# 	(x, y) = (x - pX, y - pY)
		# 	(w, h) = (w + (pX * 2), h + (pY * 2))

		# 	# extract the ROI from the image and draw a bounding box
		# 	# surrounding the MRZ
		# 	roi = image[y:y + h, x:x + w].copy()
		# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# 	print('x = ', x, 'y = ', y, 'w = ', w, 'h = ', h)
		# 	print('ratio: width/height = ', w / h)
		# 	break

		# show the output images
		cv2.imshow("Image", image)
		# cv2.imshow("ROI", roi)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
