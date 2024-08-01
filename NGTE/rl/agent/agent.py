import numpy as np

from rl.agent.core import Actor, Critic, BaseAgent, NaiveNet
import torch
import torch.nn as nn
import numpy as np
from rl.utils import net_utils

class IntCritic(nn.Module):
    def __init__(self, env_params, args, hierarchy):
        super().__init__()
        self.hierarchy = hierarchy
        self.args = args
        input_dim = env_params['obs'] + env_params['sub_goal']

        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids + [1],
            activation=args.activ)
    
    def forward(self, inputs):
        q_values = self.net(inputs)
        return q_values.squeeze()



class DistIntCritic(nn.Module):
    def __init__(self, env_params, args, hierarchy):
        super().__init__()
        self._dist_func = IntCritic(env_params, args, hierarchy)
        self.args = args
        self.gamma = args.gamma
    
    def forward(self, inputs):
        dist = self._dist_func(inputs)
        log_gamma = np.log(self.gamma)
        return -(1 - torch.exp(dist * log_gamma)) / (1 - self.gamma)
    
    def get_dist(self, inputs):
        dist = self._dist_func(inputs)
        return dist
    


class DistCritic(nn.Module):
    def __init__(self, env_params, args, hierarchy):
        super().__init__()
        self._dist_func = Critic(env_params, args, hierarchy)
        self.args = args
        self.gamma = args.gamma
    
    def forward(self, pi_inputs, actions):
        dist = self._dist_func(pi_inputs, actions)
        log_gamma = np.log(self.gamma)
        return -(1 - torch.exp(dist * log_gamma)) / (1 - self.gamma)
    
    def get_dist(self, pi_inputs, actions):
        dist = self._dist_func(pi_inputs, actions)
        if self.args.q_offset:
            dist += 1.0
        return dist
    def q_to_dist(self, q_input):
        log_gamma = np.log(self.gamma)
        dist = torch.log(1 + (1 - self.gamma) * q_input) / log_gamma
        return dist



class ExpCritic(nn.Module):
    def __init__(self, env_params, args, hierarchy):
        super().__init__()
        self._dist_func = Critic(env_params, args, hierarchy)
        self.args = args
        self.gamma = args.gamma
    
    def forward(self, pi_inputs):
        rnd_dist = self._dist_func(pi_inputs)
        log_gamma = np.log(self.gamma)
        return - (1 - torch.exp(rnd_dist * log_gamma)) / (1 - self.gamma)
    
    def get_dist(self, pi_inputs):
        dist = self._dist_func(pi_inputs)
        return dist
    
    def q_to_dist(self, q_input):
        log_gamma = np.log(self.gamma)
        dist = torch.log(1 + (1 - self.gamma) * q_input) / log_gamma
        return dist



class DistReverseCritic(nn.Module):
    def __init__(self, env_params, args, hierarchy):
        super().__init__()
        self._q_func = Critic(env_params, args, hierarchy)
        self.args = args
        self.gamma = args.gamma
    
    def forward(self, pi_inputs, actions):
        q_value = self._q_func(pi_inputs, actions)
        return q_value
    
    def get_dist(self, pi_inputs, actions):
        q_value = self._q_func(pi_inputs, actions)
        q_value = torch.clamp(q_value, min=1./(self.gamma - 1.) + 1) 
        log_gamma = np.log(self.gamma)
        dist = torch.log(1. + q_value * (1. - self.gamma)) / log_gamma
        if self.args.q_offset:
            dist += 1.0
        return dist


class LowAgent(BaseAgent):
    def __init__(self, env_params, args, name='low_agent'):
        super().__init__(env_params, args, name=name)
        self.actor = Actor(env_params, args, hierarchy='low')
        critic_func = DistCritic
        critic_int_func = DistIntCritic
        if args.use_reverse_dist_func:
            critic_func = DistReverseCritic
        self.critic1 = critic_func(env_params, args, hierarchy='low')
        self.critic2 = critic_func(env_params, args, hierarchy='low')
        self.critic1_g = critic_func(env_params, args, hierarchy='low')
        self.critic2_g = critic_func(env_params, args, hierarchy='low')
        self.critic1_int = critic_int_func(env_params, args, hierarchy='low')
        self.critic2_int = critic_int_func(env_params, args, hierarchy='low')

        self.actor_targ = Actor(env_params, args, hierarchy='low')
        self.critic1_targ = critic_func(env_params, args, hierarchy='low')
        self.critic2_targ = critic_func(env_params, args, hierarchy='low')
        self.critic1_targ_g = critic_func(env_params, args, hierarchy='low')
        self.critic2_targ_g = critic_func(env_params, args, hierarchy='low')
        self.critic1_targ_int = critic_int_func(env_params, args, hierarchy='low')
        self.critic2_targ_int = critic_int_func(env_params, args, hierarchy='low')
        
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())
        self.critic1_targ_g.load_state_dict(self.critic1_g.state_dict())
        self.critic2_targ_g.load_state_dict(self.critic2_g.state_dict())
        self.critic1_targ_int.load_state_dict(self.critic1_int.state_dict())
        self.critic2_targ_int.load_state_dict(self.critic2_int.state_dict())

        # self.naive_net = NaiveNet(env_params)

        net_utils.set_requires_grad(self.actor_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic1_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic2_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic1_targ_g, allow_grad=False)
        net_utils.set_requires_grad(self.critic2_targ_g, allow_grad=False)
        net_utils.set_requires_grad(self.critic1_targ_int, allow_grad=False)
        net_utils.set_requires_grad(self.critic2_targ_int, allow_grad=False)
        
        if self.args.cuda:
            self.cuda()
        
    def cuda(self):
        self.actor.cuda(device=self.args.cuda_num)
        self.critic1.cuda(device=self.args.cuda_num)
        self.critic2.cuda(device=self.args.cuda_num)
        self.critic1_g.cuda(device=self.args.cuda_num)
        self.critic2_g.cuda(device=self.args.cuda_num)
        self.critic1_int.cuda(device=self.args.cuda_num)
        self.critic2_int.cuda(device=self.args.cuda_num)
        self.actor_targ.cuda(device=self.args.cuda_num)
        self.critic1_targ.cuda(device=self.args.cuda_num)
        self.critic2_targ.cuda(device=self.args.cuda_num)
        self.critic1_targ_g.cuda(device=self.args.cuda_num)
        self.critic2_targ_g.cuda(device=self.args.cuda_num)
        self.critic1_targ_int.cuda(device=self.args.cuda_num)
        self.critic2_targ_int.cuda(device=self.args.cuda_num)
        # self.naive_net.cuda(device=self.args.cuda_num)
    
    @staticmethod
    def _concat(x, y):
        assert type(x) == type(y)
        if type(x) == np.ndarray:
            return np.concatenate([x, y], axis=-1)
        else:
            return torch.cat([x, y], dim=-1)
    
    def _preprocess_inputs(self, obs, goal):
        obs = self.to_2d(obs)
        goal = self.to_2d(goal)
        return obs, goal

    def _process_inputs_critic(self, obs, goal):
        return self.to_tensor(self._concat(obs, goal))
    
    def _process_inputs_actor(self, obs, goal):
        if self.args.absolute_goal:
            relative_goal = goal[:,:self.env_params['sub_goal']]
        else:
            relative_goal = goal[:,:self.env_params['sub_goal']] - obs[:,:self.env_params['sub_goal']]
        return self.to_tensor(self._concat(obs, relative_goal))
    
    def get_actions(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs_actor(obs, goal)
        with torch.no_grad():
            actions = self.actor(inputs).cpu().numpy().squeeze()
        return actions

    def _get_pairwise_dist(self, obs_tensor, ags_tensor, use_intrinsic=False, int_dist_scale=None): 
        if int_dist_scale is None:
            int_dist_scale = self.args.int_dist_scale
        batch_size = ags_tensor.shape[0]
        state_size = obs_tensor.shape[0]
        
        # Broadcasting obs_tensor to match the shape of ags_tensor
        obs_repeat_tensor = np.repeat(obs_tensor[:, None], batch_size , axis=1) # 31, batch, state dim 
        goal_repeat_tensor = np.repeat(ags_tensor[None], state_size , axis=0)
        # goal tensor = batch, sgdim
        # Preprocessing and processing the inputs
        inputs = self._process_inputs_critic(obs_repeat_tensor, goal_repeat_tensor).reshape(batch_size * state_size, -1)
        flattend_obs = obs_repeat_tensor.reshape(batch_size * state_size, -1)
        flattend_goal = goal_repeat_tensor.reshape(batch_size * state_size, -1)

        # [[2,1][3,1]] --> [[[2,1],[2,1]..]. [[3,1],[3,1]...]]
        with torch.no_grad():
            # Broadcasting the actions to match the shape of obs_repeat_tensor and goal_tensor
            actions = self.to_tensor(self.get_actions(flattend_obs, flattend_goal))
            if use_intrinsic:
                q1 = torch.clip(self.critic1_g(inputs, actions) + self.critic1_int(inputs) * self.args.int_dist_scale, self.args.g_critic_dist_clip, 1e-4)
                dist1 = self.critic1_g.q_to_dist(q1)
                q2 = torch.clip(self.critic2_g(inputs, actions) + self.critic2_int(inputs) * self.args.int_dist_scale, self.args.g_critic_dist_clip, 1e-4)
                dist2 = self.critic1_g.q_to_dist(q2)
            else:
                dist1 = self.critic1_g.get_dist(inputs, actions)
                dist2 = self.critic2_g.get_dist(inputs, actions)
            
            dist = torch.clamp(torch.maximum(dist1, dist2), min=0)

        # Squeezing the dist tensor and stacking it to form a pairwise distance matrix
        
        pairwise_dist = dist.reshape(state_size, batch_size)
        pairwise_dist = pairwise_dist.cpu().detach().numpy()
        return pairwise_dist

    
    def _get_dist_from_start(self, start, obs_tensor, use_intrinsic=False, int_dist_scale=None):
        if int_dist_scale is None:
            int_dist_scale = self.args.int_dist_scale

        start_repeat = np.ones((obs_tensor.shape[0], np.squeeze(start).shape[0])) * np.expand_dims(start, axis=0)
        obs, goal = self._preprocess_inputs(start_repeat, obs_tensor[:, :self.args.subgoal_dim])
        inputs = self._process_inputs_critic(obs, goal)
        with torch.no_grad():
            actions = self.to_tensor(self.get_actions(start_repeat, obs_tensor[:, :self.args.subgoal_dim]))
            if use_intrinsic:
                q1 = torch.clip(self.critic1_g(inputs, actions) + self.critic1_int(inputs) * int_dist_scale, self.args.g_critic_dist_clip, 1e-4)
                dist1 = self.critic1_g.q_to_dist(q1)
                q2 = torch.clip(self.critic2_g(inputs, actions) + self.critic2_int(inputs) * int_dist_scale, self.args.g_critic_dist_clip, 1e-4)
                dist2 = self.critic1_g.q_to_dist(q2)
            else:
                dist1 = self.critic1_g.get_dist(inputs, actions)
                dist2 = self.critic2_g.get_dist(inputs, actions)
            dist = torch.clamp(torch.maximum(dist1, dist2), min=0)
            
        return dist.cpu().detach().numpy()
    
    def _get_dist_to_goal(self, obs_tensor, goal, use_intrinsic=False, int_dist_scale=None): # 31,29 // batch, self.subgoal_dim
        if int_dist_scale is None:
            int_dist_scale = self.args.int_dist_scale
        batch_size = goal.shape[0]
        state_size = obs_tensor.shape[0]
        goal_repeat = np.repeat(goal[:,None], state_size, axis=1) # batch, 31, subg
        state_repeat = np.repeat(obs_tensor[None,:], batch_size, axis=0) # batch, 31, 29
        # goal_repeat = np.ones_like(obs_tensor[:, :self.args.subgoal_dim]) \
        #     * np.expand_dims(goal[:self.args.subgoal_dim], axis=0) # batch, 31, subgoal_dim
        
        inputs = self._process_inputs_critic(state_repeat, goal_repeat).reshape(batch_size * state_size, obs_tensor.shape[-1] + goal.shape[-1])
        flattened_state = state_repeat.reshape(batch_size * state_size, obs_tensor.shape[-1])
        flattened_goal = goal_repeat.reshape(batch_size * state_size, goal.shape[-1])
        with torch.no_grad():
            actions = self.to_tensor(self.get_actions(flattened_state, flattened_goal))
            if use_intrinsic:
                q1 = torch.clip(self.critic1_g(inputs, actions) + self.critic1_int(inputs) * self.args.int_dist_scale, self.args.g_critic_dist_clip, 1e-4)
                dist1 = self.critic1_g.q_to_dist(q1)
                q2 = torch.clip(self.critic2_g(inputs, actions) + self.critic2_int(inputs) * self.args.int_dist_scale, self.args.g_critic_dist_clip, 1e-4)
                dist2 = self.critic1_g.q_to_dist(q2)

            else:
                dist1 = self.critic1_g.get_dist(inputs, actions)
                dist2 = self.critic2_g.get_dist(inputs, actions)
                
            dist = torch.clip(torch.maximum(dist1, dist2), min=1)
            
        return dist.cpu().detach().numpy()

    def _get_dist_to_goal_only_nov(self, obs_tensor, goal): # 31,29 // batch, self.subgoal_dim
        if int_dist_scale is None:
            int_dist_scale = self.args.int_dist_scale
        batch_size = goal.shape[0]
        state_size = obs_tensor.shape[0]
        goal_repeat = np.repeat(goal[:,None], state_size, axis=1) # batch, 31, subg
        state_repeat = np.repeat(obs_tensor[None,:], batch_size, axis=0) # batch, 31, 29
        # goal_repeat = np.ones_like(obs_tensor[:, :self.args.subgoal_dim]) \
        #     * np.expand_dims(goal[:self.args.subgoal_dim], axis=0) # batch, 31, subgoal_dim
        
        inputs = self._process_inputs_critic(state_repeat, goal_repeat).reshape(batch_size * state_size, obs_tensor.shape[-1] + goal.shape[-1])
        flattened_state = state_repeat.reshape(batch_size * state_size, obs_tensor.shape[-1])
        flattened_goal = goal_repeat.reshape(batch_size * state_size, goal.shape[-1])
        with torch.no_grad():
            actions = self.to_tensor(self.get_actions(flattened_state, flattened_goal))
            q1 = self.critic1_int(inputs)
            dist1 = self.critic1_g.q_to_dist(q1)
            q2 = self.critic2_int(inputs)
            dist2 = self.critic1_g.q_to_dist(q2)
            
            dist = torch.clip(torch.maximum(dist1, dist2), min=1)
            
        return dist.cpu().detach().numpy()

    def _get_point_to_point(self, point1, point2, use_intrinsic=False):
        obs, goal = self._preprocess_inputs(point1, point2[:self.args.subgoal_dim])
        inputs = self._process_inputs_critic(obs, goal)
        with torch.no_grad():
            actions = torch.unsqueeze(self.to_tensor(self.get_actions(point1, point2[:self.args.subgoal_dim])), 0)
            if use_intrinsic:
                q1 = torch.clip(self.critic1_g(inputs, actions) + self.critic1_int(inputs) * self.args.int_dist_scale, self.args.g_critic_dist_clip, 1e-4)
                dist1 = self.critic1_g.q_to_dist(q1)
                q2 = torch.clip(self.critic2_g(inputs, actions) + self.critic2_int(inputs) * self.args.int_dist_scale, self.args.g_critic_dist_clip, 1e-4)
                dist2 = self.critic1_g.q_to_dist(q2)
            else:
                dist1 = self.critic1_g.get_dist(inputs, actions)
                dist2 = self.critic2_g.get_dist(inputs, actions)
            dist = torch.clamp(torch.maximum(dist1, dist2), min=0)
            
        return dist.cpu().detach().numpy()
    
    def get_pis(self, obs, goal, pi_target=False):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs_actor(obs, goal)
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        return pis
    

    def get_qs(self, obs, goal, actions, q_target=False, net=0):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs_critic(obs, goal)
        actions = self.to_tensor(actions)
        if (net == 1):
            q_net = self.critic1_targ if q_target else self.critic1
            qs = q_net(inputs, actions)
        elif (net == 2):
            q_net = self.critic2_targ if q_target else self.critic2
            qs = q_net(inputs, actions)
        elif (net == 0):
            q_net1 = self.critic1_targ if q_target else self.critic1
            q_net2 = self.critic2_targ if q_target else self.critic2
            qs1 = q_net1(inputs, actions) 
            qs2 = q_net2(inputs, actions)
            qs = torch.minimum(qs1, qs2)
        return qs

    def get_qs_g(self, obs, goal, actions, q_target=False, net=0):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs_critic(obs, goal)
        actions = self.to_tensor(actions)
        if (net == 1):
            q_net = self.critic1_targ_g if q_target else self.critic1_g
            qs = q_net(inputs, actions)
        elif (net == 2):
            q_net = self.critic2_targ_g if q_target else self.critic2_g
            qs = q_net(inputs, actions)
        elif (net == 0):
            q_net1 = self.critic1_targ_g if q_target else self.critic1_g
            q_net2 = self.critic2_targ_g if q_target else self.critic2_g
            qs1 = q_net1(inputs, actions) 
            qs2 = q_net2(inputs, actions) 
            qs = torch.minimum(qs1, qs2)
        return qs
    
    def get_qs_int(self, obs, goal, q_target=False, net=0):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs_critic(obs, goal)
        if (net == 1):
            q_net = self.critic1_targ_int if q_target else self.critic1_int
            qs = q_net(inputs)
        elif (net == 2):
            q_net = self.critic2_targ_int if q_target else self.critic2_int
            qs = q_net(inputs)
        elif (net == 0):
            q_net1 = self.critic1_targ_int if q_target else self.critic1_int
            q_net2 = self.critic2_targ_int if q_target else self.critic2_int
            qs1 = q_net1(inputs) 
            qs2 = q_net2(inputs) 
            qs = torch.minimum(qs1, qs2)
        return qs

    def forward(self, obs, goal, q_target=False, pi_target=False):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs_c = self._process_inputs_critic(obs, goal)
        inputs_a = self._process_inputs_actor(obs, goal)
        q_net1 = self.critic1_targ if q_target else self.critic1
        q_net2 = self.critic2_targ if q_target else self.critic2
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs_a)
        qs1 = q_net1(inputs_c, pis)
        qs2 = q_net2(inputs_c, pis)
        qs = torch.minimum(qs1, qs2)
        return qs, pis

    def forward1(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs_c = self._process_inputs_critic(obs, goal)
        inputs_a = self._process_inputs_actor(obs, goal)
        q_net1 = self.critic1
        a_net = self.actor
        pis = a_net(inputs_a)
        qs1 = q_net1(inputs_c, pis)
        return qs1, pis
    
    def target_update(self):
        net_utils.target_soft_update(source=self.actor, target=self.actor_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic1, target=self.critic1_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic2, target=self.critic2_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic1_g, target=self.critic1_targ_g, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic2_g, target=self.critic2_targ_g, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic1_int, target=self.critic1_targ_int, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic2_int, target=self.critic2_targ_int, polyak=self.args.polyak)
    

    
    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'actor_targ': self.actor_targ.state_dict(),
                'critic1': self.critic1.state_dict(), 'critic1_targ': self.critic1_targ.state_dict(),
                'critic2': self.critic2.state_dict(), 'critic2_targ': self.critic2_targ.state_dict(),
                'critic1_g': self.critic1_g.state_dict(), 'critic1_targ_g': self.critic1_targ_g.state_dict(),
                'critic2_g': self.critic2_g.state_dict(), 'critic2_targ_g': self.critic2_targ_g.state_dict(),
                'critic1_int': self.critic1_g.state_dict(), 'critic1_targ_int': self.critic1_targ_g.state_dict(),
                'critic2_int': self.critic2_g.state_dict(), 'critic2_targ_int': self.critic2_targ_g.state_dict()}
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_targ.load_state_dict(state_dict['actor_targ'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic1_targ.load_state_dict(state_dict['critic2_targ'])
        self.critic2.load_state_dict(state_dict['critic1'])
        self.critic2_targ.load_state_dict(state_dict['critic2_targ'])
        self.critic1_g.load_state_dict(state_dict['critic1_g'])
        self.critic1_targ_g.load_state_dict(state_dict['critic2_targ_g'])
        self.critic2_g.load_state_dict(state_dict['critic1_g'])
        self.critic2_targ_g.load_state_dict(state_dict['critic2_targ_g'])

        self.critic1_int.load_state_dict(state_dict['critic1_g'])
        self.critic1_targ_int.load_state_dict(state_dict['critic2_targ_g'])
        self.critic2_int.load_state_dict(state_dict['critic1_g'])
        self.critic2_targ_int.load_state_dict(state_dict['critic2_targ_g'])


class HighAgent(BaseAgent):
    def __init__(self, env_params, args, name='high_agent'):
        super().__init__(env_params, args, name=name)
        self.actor = Actor(env_params, args, hierarchy='high')
        critic_func = DistCritic
        if args.use_reverse_dist_func:
            critic_func = DistReverseCritic
        self.critic1 = critic_func(env_params, args, hierarchy='high')
        self.critic2 = critic_func(env_params, args, hierarchy='high')


        self.actor_targ = Actor(env_params, args, hierarchy='high')
        self.critic1_targ = critic_func(env_params, args, hierarchy='high')
        self.critic2_targ = critic_func(env_params, args, hierarchy='high')
        
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())
 
        net_utils.set_requires_grad(self.actor_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic1_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic2_targ, allow_grad=False)


        
        if self.args.cuda:
            self.cuda()
        
    def cuda(self):
        self.actor.cuda(device=self.args.cuda_num)
        self.critic1.cuda(device=self.args.cuda_num)
        self.critic2.cuda(device=self.args.cuda_num)
        self.actor_targ.cuda(device=self.args.cuda_num)
        self.critic1_targ.cuda(device=self.args.cuda_num)
        self.critic2_targ.cuda(device=self.args.cuda_num)
    
    @staticmethod
    def _concat(x, y):
        assert type(x) == type(y)
        if type(x) == np.ndarray:
            return np.concatenate([x, y], axis=-1)
        else:
            return torch.cat([x, y], dim=-1)
    
    def _preprocess_inputs(self, obs, goal):
        obs = self.to_2d(obs)
        goal = self.to_2d(goal)
        return obs, goal
    
    def _process_inputs(self, obs, goal):
        return self.to_tensor(self._concat(obs, goal))
    
    def get_actions(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        with torch.no_grad():
            actions = self.actor(inputs).cpu().numpy().squeeze()
        return actions
    
    def get_pis(self, obs, goal, pi_target=False):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        return pis
    

    def get_qs(self, obs, goal, actions, q_target=False, net=0):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        actions = self.to_tensor(actions)
        if (net == 1):
            q_net = self.critic1_targ if q_target else self.critic1
            qs = q_net(inputs, actions)
        elif (net == 2):
            q_net = self.critic2_targ if q_target else self.critic2
            qs = q_net(inputs, actions)
        elif (net == 0):
            q_net1 = self.critic1_targ if q_target else self.critic1
            q_net2 = self.critic2_targ if q_target else self.critic2
            qs1 = q_net1(inputs, actions) 
            qs2 = q_net2(inputs, actions) 
            qs = torch.minimum(qs1, qs2)
        return qs
    
    def forward(self, obs, goal, q_target=False, pi_target=False):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal) 
        q_net1 = self.critic1_targ if q_target else self.critic1
        q_net2 = self.critic2_targ if q_target else self.critic2
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        qs1 = q_net1(inputs, pis)
        qs2 = q_net2(inputs, pis)
        qs = torch.minimum(qs1, qs2)
        return qs, pis

    def forward1(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal) 
        q_net1 = self.critic1
        a_net = self.actor
        pis = a_net(inputs)
        qs1 = q_net1(inputs, pis)
        return qs1, pis
    
    def target_update(self):
        net_utils.target_soft_update(source=self.actor, target=self.actor_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic1, target=self.critic1_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic2, target=self.critic2_targ, polyak=self.args.polyak)
    

    
    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'actor_targ': self.actor_targ.state_dict(),
                'critic1': self.critic1.state_dict(), 'critic1_targ': self.critic1_targ.state_dict(),
                'critic2': self.critic2.state_dict(), 'critic2_targ': self.critic2_targ.state_dict()}
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_targ.load_state_dict(state_dict['actor_targ'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic1_targ.load_state_dict(state_dict['critic2_targ'])
        self.critic2.load_state_dict(state_dict['critic1'])
        self.critic2_targ.load_state_dict(state_dict['critic2_targ'])



class ExpHighAgent(BaseAgent):
    def __init__(self, env_params, args, name='high_agent'):
        super().__init__(env_params, args, name=name)
        self.actor = Actor(env_params, args, hierarchy='high')
        critic_func = DistCritic
        if args.use_reverse_dist_func:
            critic_func = DistReverseCritic
        self.critic1 = critic_func(env_params, args, hierarchy='high')
        self.critic2 = critic_func(env_params, args, hierarchy='high')

        self.actor_targ = Actor(env_params, args, hierarchy='high')
        self.critic1_targ = critic_func(env_params, args, hierarchy='high')
        self.critic2_targ = critic_func(env_params, args, hierarchy='high')
        
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())

        # self.naive_net = NaiveNet(env_params)

        net_utils.set_requires_grad(self.actor_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic1_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic2_targ, allow_grad=False)
        
        if self.args.cuda:
            self.cuda()
        
    def cuda(self):
        self.actor.cuda(device=self.args.cuda_num)
        self.critic1.cuda(device=self.args.cuda_num)
        self.critic2.cuda(device=self.args.cuda_num)
        self.actor_targ.cuda(device=self.args.cuda_num)
        self.critic1_targ.cuda(device=self.args.cuda_num)
        self.critic2_targ.cuda(device=self.args.cuda_num)
    
    def _preprocess_inputs(self, obs):
        obs = self.to_2d(obs)
        return obs
    
    def _process_inputs(self, obs):
        obs = self.to_tensor(obs)
        return obs
    
    def get_actions(self, obs):
        inputs = self._preprocess_inputs(obs)
        inputs = self._process_inputs(inputs)
        with torch.no_grad():
            actions = self.actor(inputs).cpu().numpy().squeeze()
        return actions

    def get_pis(self, obs, pi_target=False):
        inputs = self._preprocess_inputs(obs)
        inputs = self._process_inputs(inputs)
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        return pis
    
    def get_qs(self, obs, actions, q_target=False, net=0):
        inputs = self._preprocess_inputs(obs)
        inputs = self._process_inputs(inputs)
        actions = self.to_tensor(actions)
        if (net == 1):
            q_net = self.critic1_targ if q_target else self.critic1
            qs = q_net(inputs, actions)
        elif (net == 2):
            q_net = self.critic2_targ if q_target else self.critic2
            qs = q_net(inputs, actions)
        elif (net == 0):
            q_net1 = self.critic1_targ if q_target else self.critic1
            q_net2 = self.critic2_targ if q_target else self.critic2
            qs1 = q_net1(inputs, actions) 
            qs2 = q_net2(inputs, actions)
            qs = torch.minimum(qs1, qs2)
        return qs

    def forward(self, obs, q_target=False, pi_target=False):
        inputs = self._preprocess_inputs(obs)
        inputs = self._process_inputs(inputs)
        q_net1 = self.critic1_targ if q_target else self.critic1
        q_net2 = self.critic2_targ if q_target else self.critic2
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        qs1 = q_net1(inputs, pis)
        qs2 = q_net2(inputs, pis)
        qs = torch.minimum(qs1, qs2)
        return qs, pis

    def forward1(self, obs):
        input = self._preprocess_inputs(obs)
        input = self._process_inputs(input)
        q_net1 = self.critic1
        a_net = self.actor
        pis = a_net(input)
        qs1 = q_net1(input, pis)
        return qs1, pis
    
    def target_update(self):
        net_utils.target_soft_update(source=self.actor, target=self.actor_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic1, target=self.critic1_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic2, target=self.critic2_targ, polyak=self.args.polyak)
    
    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'actor_targ': self.actor_targ.state_dict(),
                'critic1': self.critic1.state_dict(), 'critic1_targ': self.critic1_targ.state_dict(),
                'critic2': self.critic2.state_dict(), 'critic2_targ': self.critic2_targ.state_dict(),}
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_targ.load_state_dict(state_dict['actor_targ'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic1_targ.load_state_dict(state_dict['critic2_targ'])
        self.critic2.load_state_dict(state_dict['critic1'])
        self.critic2_targ.load_state_dict(state_dict['critic2_targ'])




class ExpLowAgent(BaseAgent):
    def __init__(self, env_params, args, name='low_agent'):
        super().__init__(env_params, args, name=name)
        self.actor = Actor(env_params, args, hierarchy='low')
        critic_func = DistCritic
        if args.use_reverse_dist_func:
            critic_func = DistReverseCritic
        self.critic1 = critic_func(env_params, args, hierarchy='low')
        self.critic2 = critic_func(env_params, args, hierarchy='low')

        self.actor_targ = Actor(env_params, args, hierarchy='low')
        self.critic1_targ = critic_func(env_params, args, hierarchy='low')
        self.critic2_targ = critic_func(env_params, args, hierarchy='low')
        
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())

        # self.naive_net = NaiveNet(env_params)

        net_utils.set_requires_grad(self.actor_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic1_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic2_targ, allow_grad=False)
        
        if self.args.cuda:
            self.cuda()
        
    def cuda(self):
        self.actor.cuda(device=self.args.cuda_num)
        self.critic1.cuda(device=self.args.cuda_num)
        self.critic2.cuda(device=self.args.cuda_num)
        self.actor_targ.cuda(device=self.args.cuda_num)
        self.critic1_targ.cuda(device=self.args.cuda_num)
        self.critic2_targ.cuda(device=self.args.cuda_num)
    
    def _preprocess_inputs(self, obs):
        obs = self.to_2d(obs)
        return obs
    
    def _process_inputs(self, obs):
        obs = self.to_tensor(obs)
        return obs
    
    def get_actions(self, obs):
        inputs = self._preprocess_inputs(obs)
        inputs = self._process_inputs(inputs)
        with torch.no_grad():
            actions = self.actor(inputs).cpu().numpy().squeeze()
        return actions

    def get_pis(self, obs, pi_target=False):
        inputs = self._preprocess_inputs(obs)
        inputs = self._process_inputs(inputs)
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        return pis
    
    def get_qs(self, obs, actions, q_target=False, net=0):
        inputs = self._preprocess_inputs(obs)
        inputs = self._process_inputs(inputs)
        actions = self.to_tensor(actions)
        if (net == 1):
            q_net = self.critic1_targ if q_target else self.critic1
            qs = q_net(inputs, actions)
        elif (net == 2):
            q_net = self.critic2_targ if q_target else self.critic2
            qs = q_net(inputs, actions)
        elif (net == 0):
            q_net1 = self.critic1_targ if q_target else self.critic1
            q_net2 = self.critic2_targ if q_target else self.critic2
            qs1 = q_net1(inputs, actions) 
            qs2 = q_net2(inputs, actions)
            qs = torch.minimum(qs1, qs2)
        return qs

    def forward(self, obs, q_target=False, pi_target=False):
        inputs = self._preprocess_inputs(obs)
        inputs = self._process_inputs(inputs)
        q_net1 = self.critic1_targ if q_target else self.critic1
        q_net2 = self.critic2_targ if q_target else self.critic2
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        qs1 = q_net1(inputs, pis)
        qs2 = q_net2(inputs, pis)
        qs = torch.minimum(qs1, qs2)
        return qs, pis

    def forward1(self, obs):
        input = self._preprocess_inputs(obs)
        input = self._process_inputs(input)
        q_net1 = self.critic1
        a_net = self.actor
        pis = a_net(input)
        qs1 = q_net1(input, pis)
        return qs1, pis
    
    def target_update(self):
        net_utils.target_soft_update(source=self.actor, target=self.actor_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic1, target=self.critic1_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic2, target=self.critic2_targ, polyak=self.args.polyak)
    
    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'actor_targ': self.actor_targ.state_dict(),
                'critic1': self.critic1.state_dict(), 'critic1_targ': self.critic1_targ.state_dict(),
                'critic2': self.critic2.state_dict(), 'critic2_targ': self.critic2_targ.state_dict(),
            }
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_targ.load_state_dict(state_dict['actor_targ'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic1_targ.load_state_dict(state_dict['critic2_targ'])
        self.critic2.load_state_dict(state_dict['critic1'])
        self.critic2_targ.load_state_dict(state_dict['critic2_targ'])
