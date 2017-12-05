import random
import gym
import universe



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


def main():
    env = gym.make('internet.SlitherIO-v0')
    observation_n = env.reset()
    qTable = [[0.0 for i in range(4)] for j in range(10)]

    SOM = None  # 10 state SOM

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


    # qTable organized by [state : [action, action], state : [action, action]]


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
                    print("keys: " + universe.rewarder["reward"])
                    # update q-table with next_state &

                    som_state = random.randrange(0, 10)  # get int state by passing observation_n to SOM
                    # if is new state, append new state to qTable w/ actions
                    # SOM shouldn't be out of bounds
                    '''if som_state > len(data.qTable):
                        data.qTable.append([0.0, 0.0, 0.0])'''
                    max_reward = max(qTable[som_state])
                    index_of_reward = qTable[som_state].index(max_reward)
                    if random.uniform(0.0, 1.0) < e:
                        action_i = random.randint(0, len(action_list) - 1)
                    else:
                        action_i = index_of_reward
                else:
                    action_n = [action_list[action_i] for ob in observation_n]
            else:
                action_n = [still for ob in observation_n]

            observation_n, reward_n, done_n, info = env.step(action_n)
            reward = reward_n[0]
            #next_state = observation_n[0] # get next state from SOM
            next_state = random.randrange(0,10)
            env.render()

            #update q_table
            timestep += 1
            next_reward = max(qTable[next_state])
            new_value = (1 - alpha) * qTable[state][prev_action_i] + alpha * (beta * next_reward + reward)
            alpha = 1.0 / (1 + (timestep / 200.0))  # arbitrary constant
            e = 1.0 / (1 + (timestep / 2000.0))

            qTable[state][prev_action_i] = new_value


if __name__ == '__main__':
    main()






