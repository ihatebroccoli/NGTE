import threading
import numpy as np
import torch
import os.path as osp
import math
import time


def sample_her_transitions_with_subgoaltesting_exp(buffer, reward_func, batch_size, graphplanner, future_step, cutoff, subgoaltest_p, subgoaltest_threshold, monitor, gradual_pen, use_intrinsic, int_method, rnd, int_scale=0.2):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])

    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]  # [buffer_len, episode_len, dim]

    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    subgoaltesting_indexes = np.where(np.random.uniform(size=batch_size) < subgoaltest_p) 
    not_subgoaltesting_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'] = buffer['ag'][ep_idxes, future_t] # 도달한 것 중 t_sample + n_step로 goal relabel
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()

    int_r = np.clip(rnd.intrinsic_reward(batch['ag2']), 0., 1.0) * int_scale
    int_r_max = rnd.reward_max
    batch['r'] = int_r.copy()
    dist = batch['a'][subgoaltesting_indexes] - batch['ag2'][subgoaltesting_indexes] # ag2: next state, action relabel 안한 것으로 subgoal testing dist(next state - non relabeled subgoal)    
    dist = np.linalg.norm(dist, axis=1)
    subgoaltesting_failure = subgoaltesting_indexes[0][np.where(dist>subgoaltest_threshold)] # 실패한 subgoal testing은 페널티
    penalty = -1.3
    batch['r'][subgoaltesting_failure] = penalty
    batch['a'][not_subgoaltesting_indexes] = batch['ag2'][not_subgoaltesting_indexes] # action relabeling
    
    monitor.store(sample_high_exp_rew=int_r.mean())
    monitor.store(sample_int_r_expmax=int_r_max.mean())
    if graphplanner.graph is not None:
        dist_2 = graphplanner.dist_from_graph_to_goal(batch['a'][subgoaltesting_indexes])
        monitor.store(distance_from_graph = np.mean(dist_2))
        subgoaltesting_failure_2 = subgoaltesting_indexes[0][np.where(dist_2>(cutoff))]
        batch['r'][subgoaltesting_failure_2] = -gradual_pen

    return batch

def sample_her_transitions_exp(buffer, reward_func, batch_size, future_step, use_intrinsic, int_method, monitor, future_p=1.0, rnd=None, int_scale=0.1):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 
    
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)
    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()

    int_r = np.clip(rnd.intrinsic_reward(batch['ag2']), 0., 1.0) * int_scale
    batch['r'] =  int_r
    monitor.store(sample_low_exp_rew=int_r.mean())

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_her_transitions(buffer, reward_func, batch_size, future_step, use_intrinsic, int_method, monitor, future_p=1.0, rnd=None, int_scale=0.1):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]
    
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()} 
    
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p) 
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)
    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()

    if rnd == None:
        batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2'])
    else:
        int_r = -reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2']) * np.clip(rnd.intrinsic_reward(batch['ag2']), 0., 1.) * int_scale
        assert (int_r >= 0).any(), "negative int_r"

        batch['r'] = int_r
        monitor.store(sample_low_int_rew_int=int_r.mean())

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_her_transitions_with_subgoaltesting(buffer, reward_func, batch_size, graphplanner, future_step, cutoff, subgoaltest_p, subgoaltest_threshold, monitor, gradual_pen, use_intrinsic, int_method, rnd):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])

    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]  # [buffer_len, episode_len, dim]

    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    subgoaltesting_indexes = np.where(np.random.uniform(size=batch_size) < subgoaltest_p) 
    not_subgoaltesting_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizon - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'] = buffer['ag'][ep_idxes, future_t] # 도달한 것 중 t_sample + n_step로 goal relabel
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2']) # reward 계산
    dist = batch['a'][subgoaltesting_indexes] - batch['ag2'][subgoaltesting_indexes] # ag2: next state, action relabel 안한 것으로 subgoal testing dist(next state - non relabeled subgoal)
    batch['a'][not_subgoaltesting_indexes] = batch['ag2'][not_subgoaltesting_indexes] # action relabeling

    dist = np.linalg.norm(dist, axis=1)
    subgoaltesting_failure = subgoaltesting_indexes[0][np.where(dist>subgoaltest_threshold)] # 실패한 subgoal testing은 페널티
    not_subgoaltesting_failure_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)


    penalty = -1.3
    batch['r'][subgoaltesting_failure] = penalty

    if use_intrinsic:
        if int_method == 'graph':
            if graphplanner.states is not None:
                is_in, _ = graphplanner.goal_is_in_graph(batch['a'])
                not_in = np.delete(np.arange(batch_size), is_in)
                int_rew = (reward_func(batch['ag2'][not_in], batch['a'][not_in], None, ob=batch['o2'][not_in]) + 1) * 0.5
                batch['r'][not_in] += int_rew
                monitor.store(intrinsic_reward_high=int_rew.mean())
                
                is_in, _ = graphplanner.goal_is_in_graph(batch['a'])
                monitor.store(intrinsic_reward_action=len(is_in[is_in]) / len(batch['a']))
                batch['r'][is_in] += 0.1
                
        elif int_method == 'rnd':
            
            int_r = (reward_func(batch['ag2'], batch['a'], None, ob=batch['o2']) + 1) * np.clip(rnd.intrinsic_reward(batch['ag2']) * 500, 0, 1)
            batch['r'] += int_r
            monitor.store(sample_high_int_rew=int_r.mean())
            monitor.store(sample_high_int_reached=(reward_func(batch['ag2'], batch['a'], None, ob=batch['o2']) + 1).sum())
            
        
            


    if graphplanner.graph is not None:
        dist_2 = graphplanner.dist_from_graph_to_goal(batch['a'][subgoaltesting_indexes])
        monitor.store(distance_from_graph = np.mean(dist_2))
        subgoaltesting_failure_2 = subgoaltesting_indexes[0][np.where(dist_2>(cutoff*3))]
        batch['r'][subgoaltesting_failure_2] = -gradual_pen


    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_her_transitions_with_subgoaltesting_dynamic(buffer, horizon_buffer, reward_func, batch_size, graphplanner, future_step, cutoff, subgoaltest_p, subgoaltest_threshold, monitor, gradual_pen, use_intrinsic, int_method,  rnd):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]  # [buffer_len, transitioin_time , dim]
    n_trajs = buffer['a'].shape[0]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    horizons = horizon_buffer[ep_idxes]
    t_samples = np.random.randint(np.zeros_like(horizons), horizons)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    subgoaltesting_indexes = np.where(np.random.uniform(size=batch_size) < subgoaltest_p) 
    not_subgoaltesting_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)
    
    future_offset = (np.random.uniform(size=batch_size) * np.minimum(horizons - t_samples, future_step)).astype(int)
    future_t = (t_samples + 1 + future_offset)
    assert any(horizons >= future_t), "Something wrong"

    batch['bg'] = buffer['ag'][ep_idxes, future_t] # 도달한 것 중 t_sample + n_step로 goal relabel
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None, ob=batch['o2']) # reward 계산
    dist = batch['a'][subgoaltesting_indexes] - batch['ag2'][subgoaltesting_indexes] # ag2: next state, action relabel 안한 것으로 subgoal testing dist(next state - non relabeled subgoal)
    batch['a'][not_subgoaltesting_indexes] = batch['ag2'][not_subgoaltesting_indexes] # action relabeling

    dist = np.linalg.norm(dist, axis=1)
    subgoaltesting_failure = subgoaltesting_indexes[0][np.where(dist>subgoaltest_threshold)] # 실패한 subgoal testing은 페널티
    not_subgoaltesting_failure_indexes = np.delete(np.arange(batch_size), subgoaltesting_indexes)


    penalty = -1.3
    batch['r'][subgoaltesting_failure] = penalty

    if use_intrinsic:
        if int_method == 'graph':
            if graphplanner.states is not None:
                is_in, _ = graphplanner.goal_is_in_graph(batch['a'])
                not_in = np.delete(np.arange(batch_size), is_in)
                int_rew = (reward_func(batch['ag2'][not_in], batch['a'][not_in], None, ob=batch['o2'][not_in]) + 1) * 0.5
                batch['r'][not_in] += int_rew
                monitor.store(intrinsic_reward_high=int_rew.mean())
                
                is_in, _ = graphplanner.goal_is_in_graph(batch['a'])
                monitor.store(intrinsic_reward_action=len(is_in[is_in]) / len(batch['a']))
                batch['r'][is_in] += 0.1
                
        elif int_method == 'rnd':
            
            int_r = (reward_func(batch['ag2'], batch['a'], None, ob=batch['o2']) + 1) * np.clip(rnd.intrinsic_reward(batch['ag2']) * 100, 0, 1)
            batch['r'] += int_r
            monitor.store(sample_high_int_rew=int_r.mean())
            monitor.store(sample_high_int_reached=(reward_func(batch['ag2'], batch['a'], None, ob=batch['o2']) + 1).sum())
            
        
            


    if graphplanner.graph is not None:
        dist_2 = graphplanner.dist_from_graph_to_goal(batch['a'][subgoaltesting_indexes])
        monitor.store(distance_from_graph = np.mean(dist_2))
        subgoaltesting_failure_2 = subgoaltesting_indexes[0][np.where(dist_2>(cutoff*3))]
        batch['r'][subgoaltesting_failure_2] = -gradual_pen


    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch



def sample_transitions(buffer, batch_size):
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    return batch


class LowReplay:
    def __init__(self, env_params, args, low_reward_func, name='low_replay'):
        self.env_params = env_params
        self.args = args
        self.low_reward_func = low_reward_func
        
        self.horizon = env_params['max_timesteps']
        self.size = args.buffer_size // self.horizon
        
        self.current_size = 0
        self.n_transitions_stored = 0
        
        self.buffers = dict(ob=np.zeros((self.size, self.horizon + 1, self.env_params['obs'])),
                            ag=np.zeros((self.size, self.horizon + 1, self.env_params['sub_goal'])),
                            bg=np.zeros((self.size, self.horizon, self.env_params['sub_goal'])),
                            a=np.zeros((self.size, self.horizon, self.env_params['l_action_dim'])))
        
        self.lock = threading.Lock()
        self._save_file = str(name) + '.pt'
    
    def store(self, episodes):
        ob_list, ag_list, bg_list, a_list = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a']
        batch_size = ob_list.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers['ob'][idxs] = ob_list.copy()
            self.buffers['ag'][idxs] = ag_list.copy()
            self.buffers['bg'][idxs] = bg_list.copy()
            self.buffers['a'][idxs] = a_list.copy()
            self.n_transitions_stored += self.horizon * batch_size
    # future_step, use_intrinsic, int_method, monitor, future_p=1.0, rnd=None):
    def sample(self, batch_size, monitor, rnd):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions(temp_buffers, self.low_reward_func, batch_size,
                                             future_step=self.args.low_future_step,
                                             use_intrinsic=self.args.use_low_intrinsic,
                                             int_method=self.args.low_int_method,
                                             monitor=monitor,
                                             future_p=self.args.low_future_p,
                                             rnd=None,
                                             int_scale=self.args.int_scale,)
        return transitions

    def sample_g(self, batch_size, monitor, rnd):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions(temp_buffers, self.low_reward_func, batch_size,
                                             future_step=self.args.low_future_step,
                                             use_intrinsic=False,
                                             int_method=self.args.low_int_method,
                                             monitor=monitor,
                                             future_p=self.args.low_future_p_g,
                                             rnd=None,
                                             int_scale=self.args.int_scale,)
        return transitions
    
    def sample_int(self, batch_size, monitor, rnd):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions(temp_buffers, self.low_reward_func, batch_size,
                                             future_step=self.args.low_future_step,
                                             use_intrinsic=self.args.use_low_intrinsic,
                                             int_method=self.args.low_int_method,
                                             monitor=monitor,
                                             future_p= 0.8,
                                             rnd=rnd,
                                             int_scale=self.args.int_scale,)
        return transitions
    
    def sample_exp(self, batch_size, monitor, rnd):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions_exp(temp_buffers, self.low_reward_func, batch_size,
                                             future_step=self.args.low_future_step,
                                             use_intrinsic=True,
                                             int_method=self.args.low_int_method,
                                             monitor=monitor,
                                             future_p=self.args.low_future_p_g,
                                             rnd=rnd,
                                             int_scale=self.args.int_scale,)
        return transitions
    
    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx
    
    def get_all_data(self):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers
    
    def sample_regular_batch(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_transitions(temp_buffers, batch_size)
        return transitions
    
    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )
    
    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']
    
    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

class HighReplay:
    def __init__(self, env_params, args, high_reward_func, monitor, name='high_replay'):
        self.env_params = env_params
        self.args = args
        self.high_reward_func = high_reward_func
        self.monitor = monitor
        self.horizon = math.ceil(env_params['max_timesteps'] / args.subgoal_freq)
        
        self.size = args.buffer_size // self.horizon
        
        self.current_size = 0
        self.n_transitions_stored = 0
        
        self.buffers = dict(ob=np.zeros((self.size, self.horizon + 1, self.env_params['obs'])),
                            ag=np.zeros((self.size, self.horizon + 1, self.env_params['goal'])),
                            bg=np.zeros((self.size, self.horizon, self.env_params['goal'])),
                            a=np.zeros((self.size, self.horizon, self.env_params['h_action_dim'])))

        self.lock = threading.Lock()
        self._save_file = str(name) + '.pt'
    
    def store(self, episodes):
        ob_list, ag_list, bg_list, a_list = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a']
        batch_size = ob_list.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers['ob'][idxs] = ob_list.copy()
            self.buffers['ag'][idxs] = ag_list.copy()
            self.buffers['bg'][idxs] = bg_list.copy()
            self.buffers['a'][idxs] = a_list.copy()
            self.n_transitions_stored += self.horizon * batch_size
    
    def sample(self, batch_size, graphplanner, rnd=None) :
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]

        transitions = sample_her_transitions_with_subgoaltesting(temp_buffers, self.high_reward_func, batch_size, graphplanner,
                                             future_step=self.args.high_future_step,
                                             cutoff = self.args.cutoff,
                                             subgoaltest_p=self.args.subgoaltest_p,
                                             subgoaltest_threshold = self.args.subgoaltest_threshold,
                                             monitor = self.monitor,
                                             gradual_pen= self.args.gradual_pen,
                                             use_intrinsic = self.args.use_high_intrinsic,
                                             int_method=self.args.high_int_method,
                                             rnd=rnd,
                                             )
        return transitions
    
    def sample_exp(self, batch_size, graphplanner, rnd=None) :
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]

        transitions = sample_her_transitions_with_subgoaltesting_exp(temp_buffers, self.high_reward_func, batch_size, graphplanner,
                                             future_step=self.args.high_future_step,
                                             cutoff = self.args.cutoff,
                                             subgoaltest_p=0.2,
                                             subgoaltest_threshold = self.args.subgoaltest_threshold,
                                             monitor = self.monitor,
                                             gradual_pen= self.args.gradual_pen,
                                             use_intrinsic = self.args.use_high_intrinsic,
                                             int_method=self.args.high_int_method,
                                             rnd=rnd,
                                             int_scale=self.args.int_scale,
                                             )
        return transitions
    
    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx
    
    def get_all_data(self):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers
    
    def sample_regular_batch(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_transitions(temp_buffers, batch_size)
        return transitions
    
    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )
    
    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']
    
    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

class DynamicHighReplay:
    def __init__(self, env_params, args, high_reward_func, monitor, name='high_replay'):
        self.env_params = env_params
        self.args = args
        self.high_reward_func = high_reward_func
        self.monitor = monitor
        self.horizon = math.ceil(env_params['max_timesteps'] / args.min_subgoal_freq)
        
        self.size = args.buffer_size // self.horizon
        
        self.current_size = 0
        self.n_transitions_stored = 0
        
        self.buffers = dict(ob=np.zeros((self.size, self.horizon + 1, self.env_params['obs'])),
                            ag=np.zeros((self.size, self.horizon + 1, self.env_params['goal'])),
                            bg=np.zeros((self.size, self.horizon, self.env_params['goal'])),
                            a=np.zeros((self.size, self.horizon, self.env_params['h_action_dim'])))
        self.horizon_buffers = np.zeros(self.size,dtype=np.int16)
        self.lock = threading.Lock()
        self._save_file = str(name) + '.pt'
    
    def store(self, episodes):
        ob_list, ag_list, bg_list, a_list = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a']
        batch_size = ob_list.shape[0]
        input_horizon = bg_list.shape[1]

        if input_horizon < self.horizon + 1:
            ob_list = np.concatenate((ob_list, np.zeros([1, self.horizon - input_horizon, self.env_params['obs']])), axis=1)
            ag_list = np.concatenate((ag_list, np.zeros([1, self.horizon - input_horizon, self.env_params['goal']])), axis=1)
            bg_list = np.concatenate((bg_list, np.zeros([1, self.horizon - input_horizon, self.env_params['goal']])), axis=1)
            a_list = np.concatenate((a_list, np.zeros([1, self.horizon - input_horizon, self.env_params['h_action_dim']])), axis=1)

        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers['ob'][idxs] = ob_list.copy()
            self.buffers['ag'][idxs] = ag_list.copy()
            self.buffers['bg'][idxs] = bg_list.copy()
            self.buffers['a'][idxs] = a_list.copy()
            self.horizon_buffers[idxs] = input_horizon
            self.n_transitions_stored += self.horizon * batch_size
    
    def sample(self, batch_size, graphplanner, rnd=None) :
        temp_buffers = {}
        temp_horizon = []
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
            temp_horizon = self.horizon_buffers[:self.current_size]
        
        transitions = sample_her_transitions_with_subgoaltesting_dynamic(temp_buffers, temp_horizon, self.high_reward_func, batch_size, graphplanner,
                                             future_step=self.horizon,
                                             cutoff = self.args.cutoff,
                                             subgoaltest_p=self.args.subgoaltest_p,
                                             subgoaltest_threshold = self.args.subgoaltest_threshold,
                                             monitor = self.monitor,
                                             gradual_pen= self.args.gradual_pen,
                                             use_intrinsic = self.args.use_high_intrinsic,
                                             int_method=self.args.high_int_method,
                                             rnd=rnd,
                                             )
        return transitions
    
    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx
    
    def get_all_data(self):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        return temp_buffers
    
    def sample_regular_batch(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_transitions(temp_buffers, batch_size)
        return transitions
    
    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )
    
    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']
    
    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
