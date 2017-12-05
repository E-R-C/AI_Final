import gym
import universe
import random

import numpy as np
np.set_printoptions(threshold=np.nan)


'''
slither.io action spaces:
0 - space
1 - left
2 - right
'''



#reinforcement learning step
def determine_turn(n, turn, observation_n, j, total_sum, prev_total_sum, reward_n):
    # for every 15 obs, sum total obs and take avg
    #if (reward) lower than 0, change dir
    # if go 15 iterations & get reward each step, that's right - thats when we turn
    # 15 is arbitrary




    if j >= 15:
        # if reward == timestep (got a reward each timestep)
        if total_sum / j == 0:
            turn = True
        else:
            turn = False
        # reset vars ???
        total_sum = 0
        j = 0
        prev_total_sum = total_sum
        total_sum = 0

    else:
        turn = False

    # if have observation
    if observation_n != None:
        # inc counter & reward sum
        j += 1
        total_sum += reward_n

    return (turn, j, total_sum, prev_total_sum)


def main():

    # init universe environment
    # 2 pieces - client & remote
    # client = agent (dictates actions)
    # remove = environment (local)
    env = gym.make('internet.SlitherIO-v0')
    observation_n = env.reset()

    # init vars

    # num game iterations
    n = 0
    j = 0

    # sum of observations (compare when implement policy)
    total_sum = 0
    prev_total_sum = 0
    turn = False # dictate if going to turn

    # define keyboard actions
    # 3 each for formatting/standardization purposes
    left = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]
    right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
    forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]

    print("observation_space:{}".format(env.observation_space))
    # main logic
    while True:
        # increment a counter 4 num iterations
        n += 1

        # if at least one iteration, check if turn needed
        if n > 1:

            # check if at a turn
            if observation_n[0] != None: # if we've observed something
                # store reward in prev score
                prev_score = reward_n[0]
                print("reward: " + str(reward_n))
                #print("Observations: " + str(np.asarray(observation_n)) + "\n")
                #print("Max observation: " + str(np.asarray(observation_n).max(axis=0)))

                if turn:
                    # pick random event
                    event = random.choice([left, right])

                    # perform action
                    action_n = [event for ob in observation_n]

                    #set turn to false (bc we already turned)
                    turn = False

            elif not turn:
                # go straight if no turn needed
                action_n = [forward for ob in observation_n]


            # if observation, game started, check if turn needed
            if observation_n[0] != None:
                turn, j, total_sum, prev_total_sum = determine_turn(n, turn, observation_n[0], j, total_sum, prev_total_sum, reward_n[0])

            # save new vars for each iteration
            observation_n, reward_n, done_n, info = env.step(action_n)

            # render
            env.render()

if __name__ == '__main__':
    main()









