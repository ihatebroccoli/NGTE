from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import gymnasium
from gymnasium.envs.mujoco import mujoco_env

 
class create_pusher_v4:
    def __init__(self, seed):
        self.base_env = gymnasium.make('Pusher-v4', render_mode='rgb_array')
        self.base_env.seed = seed
        self.num_timesteps = 0
        self.action_space = self.base_env.action_space
        self.distance_threshold = 0.15
    def seed (self, seed):
        self.seed = seed
    def step(self, a):
        step_dict = {}
        observation, reward,_,_,_ = self.base_env.step(a)
        step_dict['observation'] = observation
        step_dict['achieved_goal'] = observation[17:20].copy()
        step_dict['desired_goal'] = observation[20:].copy()

        return step_dict, reward, None, None


    def reset(self):
        self.num_timesteps = 0
        reset_dict = {}
        observation = self.base_env.reset()[0]
        reset_dict['observation'] = observation
        reset_dict['achieved_goal'] = observation[17:20].copy()
        reset_dict['desired_goal'] = observation[20:].copy()
        
        return reset_dict

    def render(self, mode='rgb_array'):
        return self.base_env.render()

    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist
    
    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist