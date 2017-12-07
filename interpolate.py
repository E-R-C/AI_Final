import scipy.misc
import imageio
import datetime
#import skimage
#from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage

# import the necessary packages

import cv2

# construct the argument parse and parse the arguments

'''
lighest = 70, 60, 50

darkest = 14, 11, 10 '''



''''# load the image
im = cv2.imread("use.png", 0)
print(im.shape)
im = cv2.resize(im, (0,0), fx=0.3, fy=0.3)

im2 = cv2.medianBlur(im, 5)
#cv2.imshow("first", im)
'''
'''bg = im[:,:,0] - im[:,:,1] <= abs(25) # B == G
gr = im[:,:,1] - im[:,:,2] <= abs(25) # G == R
bitimg = np.bitwise_not(np.bitwise_and(bg, gr, dtype= np.uint8) * 255)'''
#bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
'''print(im2.shape)
#bitimg = bw[:] > 50
ret, im2 = cv2.threshold(im2, 55, 255, cv2.THRESH_BINARY)
cv2.imshow("bit", im2)
cv2.waitKey(0)'''


def supress(x):
	for f in fs:
		distx = f.pt[0] - x.pt[0]
		disty = f.pt[1] - x.pt[1]
		dist = math.sqrt(distx * distx + disty * disty)
		if (f.size > x.size) and (dist < f.size / 2):
			print(str(x.pt[0]) + " " + str(x.pt[1]))
			return True

def supress_snakes(x):
	for f in fs:
		distx = f.pt[0] - x.pt[0]
		disty = f.pt[1] - x.pt[1]
		dist = math.sqrt(distx * distx + disty * disty)
		if (f.size > x.size) and (dist < f.size / 2):
			print(str(x.pt[0]) + " " + str(x.pt[1]))
			return True



def fill_contour(orig):
	img = orig.copy()
	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2 = cv2.medianBlur(img2, 5)
	ret, img2 = cv2.threshold(img2, 65, 255, cv2.THRESH_BINARY)
	h, w, ret = img.shape
	bw_image = np.zeros((h, w, 3), np.uint8)

	# des = cv2.bitwise_not(img2)
	ret, contour, hier = cv2.findContours(img2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contour:
		if len(cnt) >= 5:
			cv2.drawContours(img2, [cnt], 0, 255, -1)
			ellipse = cv2.fitEllipse(cnt)

			(x, y), (MA, ma), angle = ellipse


			box = cv2.boxPoints(ellipse)
			box = np.int0(box)
			#if (box[])
			print(ma, MA)
			big = 0.0
			small = 0.0
			if MA > ma:
				big = MA
				small = ma
			else:
				big = ma
				small = MA
			if small / big > .8:

				cv2.ellipse(bw_image, ellipse, (255, 0, 0), cv2.FILLED)
				cv2.ellipse(img, ellipse, (0, 0, 255), 2)
		#cv2.drawContours(img, [box], 0, (0, 0, 255), cv2.FILLED)

	#kernel = np.ones((6, 6), np.uint8)
	#erosion = cv2.erode(img2, kernel, iterations=1)
	cv2.imshow('im', img)
	cv2.waitKey()

	return bw_image

orig = cv2.imread("USE3.png")
#orig = cv2.resize(orig, (0, 0), fx=0.2, fy=0.2)
img, orig = fill_contour(orig.copy())
#img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
detector = cv2.MSER_create()
fs = detector.detect(img)
fs.sort(key=lambda x: -x.size)
sfs = [x for x in fs if supress_snakes(x)]
#snakes = [[0 for i in range(5)] for j in range(5)]
h, w, ret = orig.shape
final_img = np.zeros((h, w, 3), np.uint8)

for f in sfs:
	'''width, height = orig.size().width
	new_x = (f.pt[0] / 5.0) /
	new_y = f.pt[1] / 5.0 / orig.size'''
#	print(f.)

	cv2.circle(final_img, (int(f.pt[0]), int(f.pt[1])), int(f.size / 2), (0, 255, 0), 2)



#hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in sfs]
#cv2.polylines(vis, hulls, 1, (0, 255, 0))
cv2.imshow('img', orig)
cv2.imshow('img2', img)
cv2.imshow('final', final_img)
#final_img2 = cv2.resize(final_img, (0, 0), fx=.2, fy=.2)
#cv2.imshow('img', final_img2)

cv2.waitKey()
#for f in sfs:
'''width, height = orig.size().width
	new_x = (f.pt[0] / 5.0) /
	new_y = f.pt[1] / 5.0 / orig.size'''
#	print(f.)

	#cv2(orig, (int(f.pt[0]), int(f.pt[1])), int(f.size / 2), (0, 255, 0), 2, 2)
#cv2.imshow("test", orig)
#cv2.imshow("test2", img)
#cv2.waitKey()
'''


#files = ['one_snake.png', 'use.png', 'use2.png']
#for string in files:
d_red = (65, 55, 150)
l_red = (200, 200, 250)

orig = cv2.imread('use.png')
orig = cv2.resize(orig, (0, 0), fx=0.5, fy=0.5)
img = orig.copy()
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.medianBlur(img2, 5)
ret, img2 = cv2.threshold(img2, 60, 255, cv2.THRESH_BINARY)

#des = cv2.bitwise_not(img2)
ret, contour, hier = cv2.findContours(img2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
	cv2.drawContours(img2, [cnt], 0, 255, -1)

#img2 = cv2.bitwise_not(des)

#img2 = cv2.bitwise_not(fill_img)
#img2 = im2_fill_inv | fill_img


cv2.imshow("first", img2)
cv2.waitKey()


#https://stackoverflow.com/questions/9860667/writing-robust-color-and-size-invariant-circle-detection-with-opencv-based-on
detector = cv2.MSER_create()
fs = detector.detect(img2)
fs.sort(key = lambda x: -x.size)




sfs = [x for x in fs if not supress(x)]

for f in sfs:
	cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), d_red, 2, 2)
	cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), l_red, 1, 2)

h, w = orig.shape[:2]
vis = np.zeros((h, w*2+5), np.uint8)
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
vis[:h, :w] = orig
vis[:h, w+5:w*2+5] = img
cv2.imshow("image", np.vstack([orig, img]))
cv2.waitKey()

#cv2.imshow("images", slices)
#cv2.waitKey(0)

#bwimg = cv2.cvtColor(bitimg, cv2.COLOR_BGR2GRAY)

#contours = cv2.findContours(bwimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
'''
'''circles = cv2.HoughCircles(im2, cv2.HOUGH_GRADIENT, minDist=10, 30, param1=20, param2=50, minRadius=0,maxRadius=0)
#print(len(contours))
print(circles)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
	cv2.circle(im, (i[0], i[1]), i[2], (0, 255, 0) , 2)
	cv2.circle(im, (i[0], i[1]), 2, (0, 0, 255), 3)

for c in contours:
	(x, y), r = cv2.minEnclosingCircle(c)
	center = (int(x), int(y))
	r = int(r)
	cv2.circle(bwimg, center, r, (0, 255, 0), 2)'''


#cv2.imshow("circles", im)
#cv2.waitKey(0)


#contours = cv2.findContours(slices, mode=0, method=)
#cv2.imshow("contours", contours)
#print(contours)
'''boundaries = [
	([0, 0, 0], [100, 100, 100])
]

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype="uint8")
	upper = np.array(upper, dtype="uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_not(image, image, mask=mask)

	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)

file = imageio.imread('test.jpg')

print(datetime.datetime.now().time())
newFile = scipy.misc.imresize(file, .3, interp='nearest')
print(datetime.datetime.now().time())


scipy.misc.imsave('testOutput.jpg', newFile)


snakeFile = imageio.imread('snaketest.png')
print(snakeFile)
objects = measure.label(snakeFile)
plt.imshow(objects)
plt.tight_layout()
plt.show()

from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np

n = 12
l = 256
np.random.seed(1)
im = np.zeros((l, l))
points = l * np.random.random((2, n ** 2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = filters.gaussian_filter(im, sigma= l / (4. * n))
blobs = im > 0.7 * im.mean()

all_labels = measure.label(blobs)
blobs_labels = measure.label(blobs, background=0)

plt.figure(figsize=(9, 3.5))
plt.subplot(131)
plt.imshow(blobs, cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(all_labels, cmap='spectral')
plt.axis('off')
plt.subplot(133)
plt.imshow(blobs_labels, cmap='spectral')
plt.axis('off')

plt.tight_layout()
plt.show()'''