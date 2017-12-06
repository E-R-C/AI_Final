import cv2
import universe
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import SOM

def crop_photo(array,top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    return array[top_left_y:bottom_right_y,top_left_x:bottom_right_x]

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
    cv2.imwrite("SOM.jpg", actual_array)
    # cv2.waitKey(1)


def main():
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1)
    observation_n = env.reset()
    top_left_x = 20
    top_left_y = 85
    bottom_right_x = 520
    bottom_right_y = 385
    scale_factor = 0.1
    radius = 3
    SOM_WIDTH = 4
    SOM_HEIGHT = 4
    image_height = int((bottom_right_y - top_left_y) * scale_factor)
    image_width = int((bottom_right_x - top_left_x) * scale_factor)

    som = SOM.SOM(SOM_WIDTH, SOM_HEIGHT, image_width, image_height, radius)
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
            shrunken_image = scipy.misc.imresize(photo_array,scale_factor,"nearest")
            cvImage = cv2.cvtColor(shrunken_image, cv2.COLOR_RGB2BGR)
            som.train(shrunken_image)
            cv2.imshow("RobotVision", cvImage)
            cv2.waitKey(1)
            # print(image)
            print_som(som)
            # cv2.waitKey(1)
main()

