# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

"""
Instruction: 
Currently, the following agents are `random` policy.
You need to implement the Q-learning agent, Sarsa agent and Dyna-Q agent in this file.
"""

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Sarsa agent here"""
class SarsaAgent(object):
    ##### START CODING HERE #####

    def __init__(self, num_observations, num_actions, all_actions, learning_rate = 0.05, gamma = 0.9, epsilon = 1):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros((num_observations, num_actions))

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.rand() <= self.epsilon:
            # epsilon randomly choose action
            action = np.random.choice(self.all_actions)
        else:
            # greedily choose action
            action = self.predict(observation)
        return action
    
    def learn(self, observation, action, reward, observation_, action_, done):
        """
            learn from experience
            update the Q-table
        """
        # time.sleep(0.5)

        predict_Q = self.Q[observation, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * self.Q[observation_, action_]
        self.Q[observation, action] += self.lr * (target_Q - predict_Q)

        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
    
    def predict(self, observation):
        Q_list = self.Q[observation, :]
        if np.count_nonzero(Q_list) == 0:
            action = np.random.choice(self.all_actions)
        else:
            action = np.argmax(Q_list)
        return action

    def save(self, npy_file = './Q_Table_SARSA.npy'):
        '''Save the learning process'''
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./Q_Table_SARSA.npy'):
        '''Load the learning process'''
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
    
    def set_epsilon(self, epsilon = 0.1):
        self.epsilon = epsilon

    def set_learning_rate(self, learning_rate):
        self.lr = learning_rate
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        do_something = True
        return None
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        do_something = True
        return None
    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Q-Learning agent here"""
class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, num_observations, num_actions, all_actions, learning_rate = 0.05, gamma = 0.9, epsilon = 1):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros((num_observations, num_actions))

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.rand() <= self.epsilon:
            # epsilon randomly choose action
            action = np.random.choice(self.all_actions)
        else:
            # greedily choose action
            action = self.predict(observation)
        return action
    
    def learn(self, observation, action, reward, observation_, done):
        """
            learn from experience
            update the Q-table
        """
        # time.sleep(0.5)

        predict_Q = self.Q[observation, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[observation_, :])
        self.Q[observation, action] += self.lr * (target_Q - predict_Q)

        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
    
    def predict(self, observation):
        Q_list = self.Q[observation, :]
        if np.count_nonzero(Q_list) == 0:
            action = np.random.choice(self.all_actions)
        else:
            action = np.argmax(Q_list)
        return action

    def save(self, npy_file = './Q_Table_QL.npy'):
        '''Save the learning process'''
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./Q_Table_QL.npy'):
        '''Load the learning process'''
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
    
    def set_epsilon(self, epsilon = 0.1):
        self.epsilon = epsilon

    def your_function(self, params):
        """You can add other functions as you wish."""
        do_something = True
        return None

    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Dyna-Q agent here"""
class DynaQAgent(object):
    ##### START CODING HERE #####
    def __init__(self, num_observations, num_actions, all_actions, learning_rate = 0.05, gamma = 0.9, epsilon = 1, n_steps = 10):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = {}
        self.steps  = n_steps
        self.Q = np.zeros((num_observations, num_actions))

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.rand() <= self.epsilon:
            # epsilon randomly choose action
            action = np.random.choice(self.all_actions)
        else:
            # greedily choose action
            action = self.predict(observation)
        return action
    
    def learn(self, observation, action, reward, observation_, done):
        """
            learn from experience
            update the Q-table
        """
        # time.sleep(0.5)

        predict_Q = self.Q[observation, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[observation_, :])
        self.Q[observation, action] += self.lr * (target_Q - predict_Q)

        if observation not in self.model.keys():
            self.model[observation] = {}
        self.model[observation][action] = (reward, observation_)

        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
    
    def predict(self, observation):
        Q_list = self.Q[observation, :]
        if np.count_nonzero(Q_list) == 0:
            action = np.random.choice(self.all_actions)
        else:
            action = np.argmax(Q_list)
        return action

    def exchange(self):
        for _ in range(self.steps):
            # randomly choose an state
            rand_idx = np.random.choice(range(len(self.model.keys())))
            _state = list(self.model)[rand_idx]
            # randomly choose an action
            rand_idx = np.random.choice(range(len(self.model[_state].keys())))
            _action = list(self.model[_state])[rand_idx]
            _reward, _nxtState = self.model[_state][_action]
            self.Q[_state, _action] += self.lr * (_reward + np.max(self.Q[_nxtState, :]) - self.Q[_state, _action])
    
    def save(self, npy_file = './Q_Table_DynaQ.npy'):
        '''Save the learning process'''
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self, npy_file='./Q_Table_DynaQ.npy'):
        '''Load the learning process'''
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
    
    def set_epsilon(self, epsilon = 0.1):
        self.epsilon = epsilon

    def your_function(self, params):
        """You can add other functions as you wish."""
        do_something = True
        return None

    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: (optional) Implement RL agent(s) with other exploration methods you have found"""
##### START CODING HERE #####
class RLAgentWithOtherExploration(object):
    """initialize the agent"""
    def __init__(self, all_actions):
        self.all_actions = all_actions
        self.epsilon = 1.0

    def choose_action(self, observation):
        """choose action with other exploration algorithms."""
        action = np.random.choice(self.all_actions)
        return action
    
    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #