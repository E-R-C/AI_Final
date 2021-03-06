from SOM import SOM
import cv2
import numpy as np
import gym
import universe

import math
import scipy.misc
import random
# track reward probably
import SOM

class QLearner():

    def __init__(self, width, height, action_list, random_threshold=1.0, exploration_threshold=20, decay_rate=.95, learning_rate_constant=10.0):
        self.width = width
        self.height = height
        self.action_list = action_list

        self.random_threshold = random_threshold
        self.exploration_threshold = exploration_threshold

        self.learning_rate_constant = learning_rate_constant
        self.decay_rate = decay_rate
        self.q_table = [[[0.0 for q in range(len(action_list))] for i in range(height)] for j in range(width)]
        self.learning_rate = 1.0
        self.discount_factor = .5
        self.times_chosen = [[[0 for q in range(len(action_list))] for i in range(height)] for j in range(width)]


    def select_action(self, x, y):
        saved_index = 0
        max_reward = 0

        # exploration
        if random.random() < self.random_threshold:
            return random.choice(self.action_list)

        for times_chosen in range(len(self.times_chosen[x][y])):
            if self.times_chosen[x][y][times_chosen] < self.exploration_threshold:
                return self.action_list[times_chosen]

        # exploitation
        for action_reward_index in range(len(self.q_table[x][y])):
            if self.q_table[x][y][action_reward_index] > max_reward:
                max_reward = self.q_table[x][y][action_reward_index]
                saved_index = action_reward_index


        return self.action_list[saved_index]


    def update_qtable(self, action, prev_state_w, prev_state_h, state_w, state_h, prev_reward):
        action_i = self.action_list.index(action)
        learning_rate = 1.0 / (1 + (self.times_chosen[prev_state_w][prev_state_h][action_i]) / self.learning_rate_constant) # higher the constant, higher the learning rate
        self.q_table[prev_state_w][prev_state_h][action_i] = (1.0 - learning_rate) * \
                                                             self.q_table[prev_state_w][prev_state_h][action_i] + \
                                                             learning_rate * (self.discount_factor *
                                                                              self.max_reward(state_w, state_h) + prev_reward)
        self.times_chosen[prev_state_w][prev_state_h][action_i] += 1

        self.random_threshold *= self.decay_rate

    def max_reward(self, state_w, state_h):
        return max(self.q_table[state_w][state_h])




def main():
    # setup
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1)
    observation_n = env.reset()

    trials = []
    reward_total = 0.0

    tiny_image_h = 12
    tiny_image_w = 18  # for tiny image
    SOM_WIDTH = 10
    SOM_HEIGHT = 10
    som = SOM.SOM(SOM_WIDTH, SOM_HEIGHT, tiny_image_w, tiny_image_h, radius=3)

    ## Define Actions on keyboard
    possible_actions = []
    left = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
    right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
    forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
    still = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
    possible_actions.append(left)
    possible_actions.append(right)
    possible_actions.append(forward)
    possible_actions.append(still)
    default_action_n = [still for ob in observation_n]

    q_learner = QLearner(SOM_WIDTH, SOM_HEIGHT, possible_actions)

    frame_buffer = []


    action = still
    state_w = 0
    prev_state_w = None
    prev_state_h = None

    while True:

        if observation_n[0] != None:

            frame_buffer.append(observation_n[0]["vision"])
            if len(frame_buffer) > 10:
                frame_buffer.pop(0)

            action_n = [action for ob in observation_n]
            image_for_som = process_image(observation_n[0]["vision"], tiny_image_w, tiny_image_h)
            state_w, state_h = som.train(image_for_som)
            print_som(som)
            if prev_state_w == None:
                prev_state_w = state_w
                prev_state_h = state_h
                action = q_learner.select_action(state_w, state_h)
            else:
                q_learner.update_qtable(action, prev_state_w, prev_state_h, state_w, state_h, reward_n[0])
                reward_total += reward_n[0]
                action = q_learner.select_action(state_w, state_h)
                prev_state_w = state_w
                prev_state_h = state_h
            print('reward: ' + str(reward_n[0]))

        else:
            if reward_total > 0.0:
                trials.append()
            # do stuff with death here
            action_n = [still for ob in observation_n]

        observation_n, reward_n, done_n, info = env.step(action_n)
        env.render()

def process_image(observation, tiny_image_w, tiny_image_h):
    top_left_x = 20
    top_left_y = 85
    bottom_right_x = 520
    bottom_right_y = 385
    photo_array = crop_photo(observation, top_left_x, top_left_y, bottom_right_x,
                             bottom_right_y)

    bgr = cv2.cvtColor(photo_array, cv2.COLOR_RGB2BGR)

    grayscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    filled_orbs = fill_orbs(grayscale)

    # filled_orbs = cv2.resize(filled_orbs, (0,0), fx=.5, fy=.5)
    # shrunken_image = cv2.resize(grayscale, (0, 0), fx=.5, fy=.5)

    filled_orbs = scipy.misc.imresize(filled_orbs, 0.5, "nearest")
    shrunken_image = scipy.misc.imresize(grayscale, 0.5, "nearest")

    filled_orbs = cv2.cvtColor(filled_orbs, cv2.COLOR_RGB2BGR)
    #shrunken_image = cv2.cvtColor(shrunken_image,


    tiny = cv2.resize(fill_snake(shrunken_image, filled_orbs), (tiny_image_w, tiny_image_h))
    return tiny

def crop_photo(array,top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    return array[top_left_y:bottom_right_y,top_left_x:bottom_right_x]

def fill_snake(orig, to_draw):
    img = orig.copy()
    img = cv2.medianBlur(img, 5)
    ret, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    ret, contour, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(img, [cnt], 0, 255, -1)

    detector = cv2.MSER_create()
    fs = detector.detect(img)
    fs.sort(key=lambda x: -x.size)
    sfs = [x for x in fs if not supress(x, fs)]
    h, w = orig.shape
    final_img = np.zeros((h, w, 3), np.uint8)

    for f in sfs:
        cv2.circle(to_draw, (int(f.pt[0]), int(f.pt[1])), int(f.size / 2), (0, 255, 0), cv2.FILLED)
    cv2.imshow("binary", img)
    cv2.imshow("contours", to_draw)
    return to_draw

def fill_orbs(orig):
    img = orig.copy()
    img2 = cv2.medianBlur(img, 5)
    ret, img2 = cv2.threshold(img2, 63, 255, cv2.THRESH_BINARY)
    h, w = img2.shape
    bw_image = np.zeros((h, w, 3), np.uint8)
    ret, contour, hier = cv2.findContours(img2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
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
    return bw_image


def supress(x, fs):
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

def supress_snakes(x, fs):
    for f in fs:
        distx = f.pt[0] - x.pt[0]
        disty = f.pt[1] - x.pt[1]
        dist = math.sqrt(distx * distx + disty * disty)
        if (f.size > x.size) and (dist < f.size / 2):
            return True


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
    # cv2.waitKey(1)


main()