import cv2
import universe
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import math
import SOM

def crop_photo(array,top_left_x, top_left_y, bottom_right_x, bottom_right_y):
	return array[top_left_y:bottom_right_y,top_left_x:bottom_right_x]


def supress(x, fs):
	#print(str(x.pt[0]) + " " + str(x.pt[1]))
	# leaderboard
	if (x.pt[0] > 165.0 and x.pt[0] < 208.0 and x.pt[1] < 15.0):
		return True
	# mini map
	if (x.pt[0] > 195.0 and x.pt[0] < 230.0 and x.pt[1] < 120.0 and x.pt[1] > 98.0):
		return True
	for f in fs:
		distx = f.pt[0] - x.pt[0]
		disty = f.pt[1] - x.pt[1]
		dist = math.sqrt(distx * distx + disty * disty)

		if (f.size > x.size) and (dist < f.size / 2):
			return True

def fill_contour(orig):
	img = orig.copy()
	img2 = cv2.medianBlur(img, 5)
	ret, img2 = cv2.threshold(img2, 60, 255, cv2.THRESH_BINARY)

	# des = cv2.bitwise_not(img2)
	ret, contour, hier = cv2.findContours(img2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contour:
		cv2.drawContours(img2, [cnt], 0, 255, -1)


	detector = cv2.MSER_create()
	fs = detector.detect(img2)
	fs.sort(key=lambda x: -x.size)
	sfs = [x for x in fs if not supress(x, fs)]
	# snakes = [[0 for i in range(5)] for j in range(5)]
	h, w = orig.shape
	final_img = np.zeros((h, w, 3), np.uint8)

	for f in sfs:
		'''width, height = orig.size().width
		new_x = (f.pt[0] / 5.0) /
		new_y = f.pt[1] / 5.0 / orig.size'''
		#	print(f.)

		cv2.circle(final_img, (int(f.pt[0]), int(f.pt[1])), int(f.size / 2), (0, 255, 0), cv2.FILLED)
		#print(str(f.pt[0]) + " " + str(f.pt[1]))
		# 173.56898498535156 3.4038755893707275
		# 186.07723999023438 3.6954381465911865
		# 192.65721130371094 3.749513626098633

		# 172.2945556640625 8.26038932800293
		# 225.9551239013672 105.84054565429688
		# 205.7675018310547 123.21642303466797
		# 204.9099578857422 122.57762908935547
		#cv2.circle(final_img, (225, 105), 25, (0, 255, 0), cv2.FILLED)
	#cv2.imshow("binary", img2)
	#cv2.imshow("contours", final_img)
	return final_img


def find_orbs(orig):

	img = orig.copy()
	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2 = cv2.medianBlur(img2, 5)
	ret, img2 = cv2.threshold(img2, 63, 255, cv2.THRESH_BINARY)
	img3 = img2.copy()
	cv2.imshow('thresholded', img3)
	h, w = img2.shape
	bw_image = np.zeros((h, w, 3), np.uint8)
	#kernel = np.ones((5,5), np.uint8)
	#img2 = cv2.erode(img, kernel, iterations=1)
	# des = cv2.bitwise_not(img2)
	ret, contour, hier = cv2.findContours(img2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contour:
		if len(cnt) >= 5:
			ellipse = cv2.fitEllipse(cnt)
			(x, y), (MA, ma), angle = ellipse


			box = cv2.boxPoints(ellipse)
			box = np.int0(box)
			#if (box[])
			big = 0.0
			small = 0.0
			if MA > ma:
				big = MA
				small = ma
			else:
				big = ma
				small = MA
			if small / big > .7 and (math.pi * MA * ma <= 200 and math.pi * MA * ma >= 20):

				cv2.ellipse(bw_image, ellipse, (255, 0, 0), cv2.FILLED)
				cv2.ellipse(img, ellipse, (0, 0, 255), 2)
		#cv2.drawContours(img, [box], 0, (0, 0, 255), cv2.FILLED)

	#kernel = np.ones((6, 6), np.uint8)
	#erosion = cv2.erode(img2, kernel, iterations=1)
	#cv2.rectangle(bw_image, pt1)
	cv2.imshow('im', img)
	cv2.imshow('eroded', img2)
	cv2.imshow('b&bimg', bw_image)

	return bw_image



def print_som(som):
	columns = []
	for w in range(som.width):
		array = []
		for h in range(som.height):
			array.append(som.get(w,h).array)
		column = np.vstack(array)
		columns.append(column)
	actual_array = np.hstack(columns)
	# print(actual_array)
	cv2.imshow("SOM.jpg", actual_array)
	cv2.waitKey(5)


def main():
	env = gym.make('internet.SlitherIO-v0')
	env.configure(remotes=1)
	observation_n = env.reset()
	top_left_x = 20
	top_left_y = 85
	tiny_image_h = 12
	tiny_image_w = 18 # for tiny image
	SOM_WIDTH = 6
	SOM_HEIGHT = 6
	som = SOM.SOM(SOM_WIDTH, SOM_HEIGHT, tiny_image_w, tiny_image_h, radius=3)

	bottom_right_x = 520
	bottom_right_y = 385

	## Define Actions on keyboard
	possible_actions = []
	left = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
	right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
	forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
	possible_actions.append(left)
	possible_actions.append(right)
	possible_actions.append(forward)

	## Setup SOM
	default_action_n = [forward for ob in observation_n]
	# main logic
	while True:
		# increment a counter for number of iterations
		observation_n, reward_n, done_n, info = env.step(default_action_n)
		env.render()
		if observation_n[0] != None:
			photo_array = crop_photo(observation_n[0]["vision"], top_left_x, top_left_y, bottom_right_x,
									 bottom_right_y)
			#shrunken_image = scipy.misc.imresize(photo_array, 0.5, "nearest")
			shrunken_image = cv2.cvtColor(photo_array, cv2.COLOR_RGB2BGR)
			#shrunken_image2 = cv2.cvtColor(shrunken_image, cv2.COLOR_BGR2GRAY)

			tiny = find_orbs(shrunken_image)

			tiny = cv2.resize(tiny, (18, 12))
			som.train(tiny)

			cv2.imshow("RobotVision", shrunken_image)
			print_som(som)
			cv2.waitKey(1)

main()