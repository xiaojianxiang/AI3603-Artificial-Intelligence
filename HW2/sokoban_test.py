import math, os, time, sys
import numpy as np
import random
import gym
from gym.wrappers import Monitor
from gym_gridworld import CliffWalk
import gym_sokoban
from agent import *

"""
1. This is an example to record video with gym library.
2. You can choose other video record tools as you wish. 
3. You DON'T NEED to upload this file in your assignment.
"""

# record sokoban environment
# the video file will be saved at the ./video folder.
env = Monitor(gym.make('Sokoban-hw2-v0'), './video', force=True)

num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# get the size of observation space 
num_observations = 1000000
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

agent = QLearningAgent(num_observations, num_actions, all_actions)
Q_QL = np.load('/Users/xiaojian_xiang/Projects/AI3606/HW2/T2_QL/Q_Table_zero.npy')

def obs_to_state(list):
    key = list[0] + list[1] * 10 + list[2] * 100 + list[3] * 1000 + list[4] * 10000 + list[5] * 100000
    return key

s = env.reset()
s_key = obs_to_state(s)
env.render()
while True:
    a = np.argmax(Q_QL[s_key, :])
    s_, r, isdone, info = env.step(a)
    env.render()
    time.sleep(1)
    s = s_
    s_key = obs_to_state(s)
    if isdone:
        break

# the recorder will stop when calling `env.close()` function
# the video file .mp4 will be saved at the ./video folder
env.close()


