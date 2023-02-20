# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 22:01:22 2023

@author: vinit
"""

import numpy as np

# Define the number of states and actions
num_states = 50
num_actions = 4
destination = 49

# Initialize the Q-table
Q = np.zeros((num_states, num_actions))

# Define the learning rate and discount factor
alpha = 0.1
gamma = 0.9

# Define the reward function
def get_reward(state, action, next_state):
    # If the robot reaches the exit, return a large positive reward
    if next_state == destination:
        return 100
    # If the robot hits an obstacle, return a negative reward
    if action == 0 and next_state in [2, 3, 5, 9, 15, 20, 21, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]:
        return -100
    # Otherwise, return a small positive reward
    return 1

# Define the update rule for the Q-table
def update_Q(state, action, next_state, reward, Q, alpha, gamma):
    max_Q = max(Q[next_state, :])
    Q[state, action] = Q[state, action] + alpha * (reward + gamma * max_Q - Q[state, action])
    return Q

# Start the training loop
for i in range(10000):
    # Initialize the state and action
    state = 0
    action = np.argmax(Q[state, :] + np.random.randn(1, num_actions) * (1. / (i + 1)))
    while state != destination:
        # Take the action and observe the next state and reward
        next_state = state + 1
        reward = get_reward(state, action, next_state)
        # Update the Q-table
        Q = update_Q(state, action, next_state, reward, Q, alpha, gamma)
        # Update the state and action
        state = next_state
        action = np.argmax(Q[state, :] + np.random.randn(1, num_actions) * (1. / (i + 1)))
print(Q)