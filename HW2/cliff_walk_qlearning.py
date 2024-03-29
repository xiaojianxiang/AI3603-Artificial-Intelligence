# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from gym_gridworld import CliffWalk
from agent import QLearningAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# construct the environment
env = CliffWalk()
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# get the size of observation space 
num_observations = env.observation_space.n
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

##### START CODING HERE #####

# construct the intelligent agent.
agent = QLearningAgent(num_observations, num_actions, all_actions)

# agent.restore('/Users/xiaojian_xiang/Projects/AI3606/HW2/QLearning/Q_Table_QL.npy')

# The scope of the episode reward
episode_rewards = []
episode_epsilons = []
# epsilon = 0.95
epsilon = 1.0

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    env.render()
    # agent interacts with the environment
    # episode_epsilons.append(epsilon+0.05)
    # agent.set_epsilon(epsilon+0.05)
    episode_epsilons.append(epsilon)
    agent.set_epsilon(epsilon)
    # Set epsilon = epsilon(k-1) * 0.95
    for iter in range(500):
        # choose an action
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        env.render()
        # update the episode reward
        episode_reward += r
        print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience
        agent.learn(s, a, r, s_, isdone)
        s = s_
        if isdone:
            # time.sleep(0.5)
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)
    episode_rewards.append(episode_reward)
    epsilon = epsilon * 0.95
    agent.save('/Users/xiaojian_xiang/Projects/AI3606/HW2/QLearning/Q_Table_zero.npy')

# Save the episode reward
np.save('/Users/xiaojian_xiang/Projects/AI3606/HW2/QLearning/reward_zero.npy', episode_rewards)
# Save the episode epsilon
np.save('/Users/xiaojian_xiang/Projects/AI3606/HW2/QLearning/epsilon_zero.npy', episode_epsilons)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Rewards')
plt.savefig('/Users/xiaojian_xiang/Projects/AI3606/HW2/QLearning/Cliff_reward_zero.png')
plt.show()

plt.figure(2)
plt.plot(episode_epsilons)
plt.xlabel('Episode')
plt.ylabel('Episode Epsilons')
plt.savefig('/Users/xiaojian_xiang/Projects/AI3606/HW2/QLearning/Cliff_epsilon_zero.png')
plt.show()

print('\ntraining over\n')   

# close the render window after training.
env.close()

##### START CODING HERE #####


