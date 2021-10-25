# -*- coding:utf-8 -*-
# Train Sarsa in Sokoban environment
import math, os, time, sys
import pdb
import numpy as np
import random, gym
from gym.wrappers import Monitor
from agent import SarsaAgent
import gym_sokoban
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####


# construct the environment
env = gym.make('Sokoban-hw2-v0')
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# get the size of observation space 
num_observations = 1000000
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 


####### START CODING HERE #######
def obs_to_state(list):
    key = list[0] + list[1] * 10 + list[2] * 100 + list[3] * 1000 + list[4] * 10000 + list[5] * 100000
    return key

# construct the intelligent agent.
agent = SarsaAgent(num_observations, num_actions, all_actions)

# The scope of the episode reward
episode_rewards = []
episode_epsilons = []
# epsilon = 0.95
epsilon = 1.0

# start training
for episode in range(1000):
    episode_reward = 0
    s = env.reset()
    s_key = obs_to_state(s)
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    env.render()
    # episode_epsilons.append(epsilon + 0.05)
    # agent.set_epsilon(epsilon + 0.05)
    episode_epsilons.append(epsilon)
    agent.set_epsilon(epsilon)
    # Set epsilon = epsilon(k-1) * 0.95

    a = agent.choose_action(s_key)
    # agent interacts with the environment
    for iter in range(500):
        s_, r, isdone, info = env.step(a)
        s_key_ = obs_to_state(s_)
        a_ = agent.choose_action(s_key_)
        env.render()
        episode_reward += r
        print(f"{s} {a} {s_} {r} {isdone}")
        agent.learn(s_key, a, r, s_key_, a_, isdone)
        s = s_
        a = a_
        s_key = s_key_
        if isdone:
            #time.sleep(0.5)
            break

    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon) 
    episode_rewards.append(episode_reward)
    epsilon = epsilon * 0.95 
    agent.save('/Users/xiaojian_xiang/Projects/AI3606/HW2/T2_SA/Q_Table_zero.npy')

# Save the episode reward
np.save('/Users/xiaojian_xiang/Projects/AI3606/HW2/T2_SA/reward_zero.npy', episode_rewards)
# Save the episode epsilon
np.save('/Users/xiaojian_xiang/Projects/AI3606/HW2/T2_SA/epsilon_zero.npy', episode_epsilons)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Rewards')
plt.savefig('/Users/xiaojian_xiang/Projects/AI3606/HW2/T2_SA/SO_reward_zero.png')
plt.show()

plt.figure(2)
plt.plot(episode_epsilons)
plt.xlabel('Episode')
plt.ylabel('Episode Epsilons')
plt.savefig('/Users/xiaojian_xiang/Projects/AI3606/HW2/T2_SA/SO_epsilon_zero.png')
plt.show()

print('\ntraining over\n')   

# close the render window after training.
env.close()

####### START CODING HERE #######





