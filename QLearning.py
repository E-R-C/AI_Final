import random
import gym
import universe

from SOM import SOM
import cv2
import numpy as np
import scipy.misc

'''def determine_action(observation_n, reward, data):
	# determine state using SOM using observation_n
	# determine reward of current state
	som_state = 0 # get int state by passing observation_n to SOM
	# if is new state, append new state to qTable w/ actions
	# SOM shouldn't be out of bounds
	if som_state > len(data.qTable):
		data.qTable.append([0.0, 0.0, 0.0])
	max_reward = max(data.qTable[som_state])
	index_of_reward = data.qTable[som_state].index(max_reward)

	if random.uniform(0.0, 1.0) < data.e:
		action = random.choice(data.action_list)
	else:
		action = data.action_list[index_of_reward]

	return action'''





	# at the next step forward, do q_update
	# env.step() ? to return






'''def q_update(next_state, reward, data): # int, int
	# inc timestep
	data.timestep += 1
	next_reward = max(data.qTable[next_state])
	new_value = (1 - data.alpha) * data.qTable[data.state][data.action] + data.alpha * (data.beta * next_reward + reward)
	data.state = next_state
	data.alpha = 1.0 / (1 + (data.timestep / 200.0)) #arbitrary constant
	e = 1.0 / (1 + (data.timestep / 80.0))
	return new_value'''

from universe import wrappers
def crop_photo(array,top_left_x, top_left_y, bottom_right_x, bottom_right_y):
	return array[top_left_y:bottom_right_y,top_left_x:bottom_right_x]


#def qUpdate(current_state, prev_state)

def fill_contour(orig):
	img = orig.copy()
	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2 = cv2.medianBlur(img2, 5)
	ret, img2 = cv2.threshold(img2, 60, 255, cv2.THRESH_BINARY)

	# des = cv2.bitwise_not(img2)
	ret, contour, hier = cv2.findContours(img2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contour:
		cv2.drawContours(img2, [cnt], 0, 255, -1)
	cv2.imshow("contours", img2)
	return img2


def main():
	env = gym.make('internet.SlitherIO-v0')
	env.configure(remotes=1)
	observation_n = env.reset()
	qTable = [[0.0 for i in range(4)] for j in range(100)]
	som_h = 10
	som_w = 10
	Som = SOM(height=som_h, width=som_w, image_height= 90, image_width=150, radius=3)  # 10 state SOM
	#SomGrd = [[0.0 for j in range(som_h)] for i in range(som_w)]
	#data = Data(SOM, qTable)
	state = 0  # might have to turn into int
	reward = 0.0
	alpha = 0.0
	beta = 0.0
	action_i = 0
	prev_action_i = 0
	timestep = 0
	temp_timestep = 0
	e = 1.0 # prob of taking random action
	left = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
	right = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
	forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
	still = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]

	action_list = [right, left, forward, still]
	reward_track = np.array([0, 0, 0, 0, 0, 0])


	# qTable organized by [state : [action, action], state : [action, action]]
	#print(env2.reset())

	action_i = random.randint(0, len(action_list) - 1)

	# determine next action
	while True:
		timestep += 1
		if timestep > 1:
			temp_timestep += 1
			if temp_timestep > 5:
				temp_timestep = 0
				if observation_n[0] != None:

					action_n = [action_list[action_i] for ob in observation_n]
					prev_action_i = action_i

					# take step w/ action
					# universe.rewarder.remote? universe
					# update q-table with next_state &
					shrunken_image = fix_image(observation_n[0]["vision"])
					(x, y) = Som.train(shrunken_image)  # get int state by passing observation_n to SOM
					next_state = y * som_w + x
					# if is new state, append new state to qTable w/ actions
					# SOM shouldn't be out of bounds
					'''if som_state > len(data.qTable):
						data.qTable.append([0.0, 0.0, 0.0])'''
					max_reward = max(qTable[next_state]) + reward_track.sum()
					for index in range(len(reward_track), 0, -1):
						reward_track[index] = reward_track[index - 1]
					reward_track[-1] = reward_n[0]
					index_of_reward = qTable[next_state].index(max_reward)
					if random.uniform(0.0, 1.0) < e:
						action_i = random.randint(0, len(action_list) - 1)
					else:
						action_i = index_of_reward
				else:
					action_n = [action_list[action_i] for ob in observation_n]
			else:
				action_n = [still for ob in observation_n]

			observation_n, reward_n, done_n, info = env.step(action_n)
			if (info['n'][0]['stats.reward.count'] != 0):
				print(info['n'][0]['stats.reward.count'])

			reward = reward_n[0]

			#next_state = observation_n[0] # get next state from SOM
			if observation_n[0] != None:
				#update q table
				shrunken_image = fix_image(observation_n[0]["vision"])
				(x, y) = Som.train(shrunken_image)
				next_state = y * som_w + x

				timestep += 1
				next_reward = max(qTable[next_state])
				new_value = (1 - alpha) * qTable[state][prev_action_i] + alpha * (beta * next_reward + reward)
				alpha = 1.0 / (1 + (timestep / 200.0))  # arbitrary constant
				e = 1.0 / (1 + (timestep / 2000.0))

				qTable[state][prev_action_i] = new_value
				state = next_state


			env.render()












def fix_image(img):
	top_left_x = 20
	top_left_y = 85
	bottom_right_x = 520
	bottom_right_y = 385
	photo_array = crop_photo(img, top_left_x, top_left_y, bottom_right_x,
							 bottom_right_y)
	shrunken_image = scipy.misc.imresize(photo_array, 0.3, "nearest")
	cvImage = cv2.cvtColor(shrunken_image, cv2.COLOR_RGB2BGR)

	cv2.imshow("RobotVision", shrunken_image)
	cv2.waitKey(1)
	return shrunken_image

if __name__ == '__main__':
	main()






