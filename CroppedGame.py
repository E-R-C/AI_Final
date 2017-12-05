import cv2
import universe
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

def crop_photo(array,top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    return array[top_left_y:bottom_right_y,top_left_x:bottom_right_x]

def main():
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1)
    observation_n = env.reset()
    top_left_x = 20
    top_left_y = 85
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
            photo_array = crop_photo(observation_n[0]["vision"], top_left_x,top_left_y,bottom_right_x,bottom_right_y)
            shrunken_image = scipy.misc.imresize(photo_array,0.3,"nearest")
            cvImage = cv2.cvtColor(shrunken_image, cv2.COLOR_RGB2BGR)

            cv2.imshow("RobotVision", cvImage)
            cv2.waitKey(1)
main()