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

    def __init__(self, width, height, action_list, random_threshold=1.0, exploration_threshold=20, decay_rate=.95, learning_rate_constant=10.0, random_floor=0.05):
        self.width = width
        self.height = height
        self.action_list = action_list

        self.random_threshold = random_threshold
        self.exploration_threshold = exploration_threshold
        self.random_floor = random_floor

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

        if self.random_threshold == self.random_floor or self.random_threshold < self.random_floor:
            self.random_threshold = self.random_floor
        else:
            self.random_threshold *= self.decay_rate

    def max_reward(self, state_w, state_h):
        return max(self.q_table[state_w][state_h])

def main():
    # setup
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1, fps=5)
    observation_n = env.reset()


    tiny_image_h = 12
    tiny_image_w = 18  # for tiny image
    SOM_WIDTH = 10
    SOM_HEIGHT = 10
    buffer_threshold = 10
    som = SOM.SOM(SOM_WIDTH, SOM_HEIGHT, tiny_image_w, tiny_image_h, learning_rate=1, decay_rate=.95, radius=3)

    ## Define Actions on keyboard
    possible_actions = []
    left_boost = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
    left = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
    right = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
    right_boost = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
    forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
    still = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
    possible_actions.append(still)
    possible_actions.append(forward)
    possible_actions.append(left)
    possible_actions.append(right)
    possible_actions.append(right_boost)
    possible_actions.append(left_boost)
    default_action_n = [still for ob in observation_n]
    bad_actions = [left_boost, right_boost, forward]

    q_learner = QLearner(SOM_WIDTH, SOM_HEIGHT, possible_actions)

    frame_buffer = []
    action_buffer = []
    rewards_buffer = []


    action = still
    state_w = 0
    prev_state_w = None
    prev_state_h = None
    last_image = None
    while True:

        if observation_n[0] != None :
            if info['n'][0]["env_status.env_state"] == "running" and reward_n[0] is not None:
                frame_buffer.append(crop(observation_n[0]["vision"]))
                action_buffer.append(action)
                rewards_buffer.append(reward_n[0])
                if len(frame_buffer) > buffer_threshold:
                    current_frame = frame_buffer.pop(0)
                    current_action = action_buffer.pop(0)
                    current_reward = rewards_buffer.pop(0)
                    action_n = [action for ob in observation_n]
                    last_image = current_frame
                    image_for_som = process_image(last_image, tiny_image_w, tiny_image_h)
                    state_w, state_h = som.train(image_for_som)
                    if prev_state_w is None:
                        prev_state_w = state_w
                        prev_state_h = state_h
                        action = q_learner.select_action(state_w, state_h)
                    else:
                        default_reward = 1
                        if current_action in bad_actions:
                            if current_reward < 0:
                                default_reward = current_reward
                            else:
                                default_reward = 0
                        q_learner.update_qtable(current_action, prev_state_w, prev_state_h, state_w, state_h, default_reward)
                        action = q_learner.select_action(state_w, state_h)
                        prev_state_w = state_w
                        prev_state_h = state_h
                    cv2.imshow("Frame Training", last_image)
        else:
            # do stuff with death here
            if len(frame_buffer) > 0:
                current_frame = frame_buffer.pop(0)
                current_action = action_buffer.pop(0)
                current_reward = rewards_buffer.pop(0)
                last_image = current_frame
                image_for_som = process_image(last_image, tiny_image_w, tiny_image_h)
                state_w, state_h = som.train(image_for_som)
                # The following assumes that prev_state has been defined
                q_learner.update_qtable(current_action, prev_state_w, prev_state_h, state_w, state_h, -500)
                action = q_learner.select_action(state_w, state_h)
                prev_state_w = state_w
                prev_state_h = state_h
                cv2.imshow("Frame Training", last_image)
            action_n = [still for ob in observation_n]
        observation_n, reward_n, done_n, info = env.step(action_n)
        # print(done_n)

        print_som(som)
        env.render()


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
    enlarged_image = scipy.misc.imresize(actual_array,200,"cubic")
    cv2.imshow("SOM.jpg", enlarged_image)
    cv2.waitKey(5)
def crop(observation):
    top_left_x = 20
    top_left_y = 85
    bottom_right_x = 520
    bottom_right_y = 385
    photo_array = crop_photo(observation, top_left_x, top_left_y, bottom_right_x,
                             bottom_right_y)
    return photo_array
def process_image(observation, tiny_image_w, tiny_image_h):
    shrunken_image = scipy.misc.imresize(observation, 0.5, "nearest")
    shrunken_image = cv2.cvtColor(shrunken_image, cv2.COLOR_RGB2BGR)
    shrunken_image2 = cv2.cvtColor(shrunken_image, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(fill_contour(shrunken_image2), (tiny_image_w, tiny_image_h))
    return tiny

def crop_photo(array,top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    return array[top_left_y:bottom_right_y,top_left_x:bottom_right_x]

def fill_contour(orig):
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
        cv2.circle(final_img, (int(f.pt[0]), int(f.pt[1])), int(f.size / 2), (0, 255, 0), cv2.FILLED)
    #cv2.imshow("binary", img2)
    #cv2.imshow("contours", final_img)
    return final_img

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

main()