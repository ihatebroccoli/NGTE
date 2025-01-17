from .reacher import Reacher3DEnv
from .pusher import PusherEnv
from collections import OrderedDict
import gym
import numpy as np
from gym import Wrapper
from gym.envs.registration import EnvSpec


class GoalWrapper(Wrapper):
    def __init__(self, env, env_name, reward_shaping='dense', seed=0, subgoal_repr='subspace', mask_goal_in_obs=False):
        super(GoalWrapper, self).__init__(env)
        self.env_name = env_name
        ob_space = env.observation_space
        high = np.array([np.inf, np.inf, np.inf])
        low = -high
        goal_space = gym.spaces.Box(low=low, high=high)

        if subgoal_repr == 'subspace':
            achieved_goal_space = goal_space
        elif subgoal_repr == 'whole':
            achieved_goal_space = ob_space
        else:
            raise NotImplementedError
        self.subgoal_repr = subgoal_repr

        self.observation_space = gym.spaces.Dict(OrderedDict({
            'observation': ob_space,
            'desired_goal': goal_space,
            'achieved_goal': achieved_goal_space,
        }))

        self.distance_threshold = 0.25
        self.reward_shaping = reward_shaping
        self.mask_goal_in_obs = mask_goal_in_obs

    def step(self, action):
        obs, sparse_reward, done, info = self.env.step(action)
        if self.env_name == "Reacher3D-v0":
            achieved_goal = self.env.get_EE_pos(obs[None]).squeeze()
        elif self.env_name == "Pusher-v0":
            achieved_goal = self.env.ac_goal_pos
        else:
            raise NotImplementedError

        if self.mask_goal_in_obs:
            obs[7:10] = 0.

        out = {
            'observation': obs,
            'desired_goal': self.env.goal,
            'achieved_goal': achieved_goal}

        if self.reward_shaping == 'dense':
            reward = -np.sum(np.square(achieved_goal - self.env.goal))
            reward -= 0.0001 * np.square(action).sum()
        elif self.reward_shaping == 'sparse':
            reward = self.compute_reward(achieved_goal, self.env.goal ,sparse=True)
        else:
            raise NotImplementedError


        return out, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.env_name == "Reacher3D-v0":
            achieved_goal = self.env.get_EE_pos(obs[None]).squeeze()
        elif self.env_name == "Pusher-v0":
            achieved_goal = self.env.ac_goal_pos
        else:
            raise NotImplementedError

        if self.mask_goal_in_obs:
            obs[7:10] = 0.

        out = {
            'observation': obs,
            'desired_goal': self.env.goal,
            'achieved_goal': achieved_goal}

        return out

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def high_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False) * 0.5

    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist



def create_fetch_env(env_name=None, seed=0, reward_shaping='sparse', subgoal_repr='subspace', mask_goal_in_obs=False):
    if env_name == "Reacher3D-v0":
        cls = Reacher3DEnv
    elif env_name == "Pusher-v0":
        cls = PusherEnv
    else:
        raise NotImplementedError

    """
    gym_mujoco_kwargs = {
        'seed': seed,
    }
    gym_env = cls(**gym_mujoco_kwargs)
    """
    gym_env = cls()
    gym_env.reset()
    return GoalWrapper(gym_env, env_name, reward_shaping=reward_shaping, seed=seed, subgoal_repr=subgoal_repr,
                       mask_goal_in_obs=mask_goal_in_obs)