import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rl.algo.core import BaseAlgo
from rl.algo.graph import GraphPlanner
from rl.rnd.rnd import RND
import time
import os
from rl.agent.agent import ExpLowAgent, ExpHighAgent
from rl.learn.ngte import ExpLowLearner, ExpHighLearner

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import moviepy

class Algo(BaseAlgo):
    def __init__(
        self,
        env, env_params, args,
        test_env,
        low_agent, high_agent, low_replay, high_replay, monitor, low_learner, high_learner,
        low_reward_func, high_reward_func,
        name='algo',
    ):
        super().__init__(
            env, env_params, args,
            low_agent, high_agent, low_replay, high_replay, monitor, low_learner, high_learner,
            low_reward_func, high_reward_func,
            name=name,
        )
        self.test_env = test_env
        self.fps_landmarks = None
        self.curr_subgoal = None
        self.is_reachable = False
        self.curr_high_act = None
        self.curr_highpolicy_obs = None
        self.way_to_subgoal = 0
        self.subgoal_freq = args.subgoal_freq
        self.subgoal_scale = np.array(args.subgoal_scale)
        self.subgoal_offset = np.array(args.subgoal_offset)
        self.subgoal_dim = args.subgoal_dim
        self.low_replay = low_replay
        self.high_replay = high_replay          
        self.graphplanner = GraphPlanner(args, low_replay, low_agent, env)
        self.waypoint_subgoal = None
        self.step_count_for_dynamic = 0
        self.env_frames = []
        if args.rnd:
            self.rnd = RND(self.subgoal_dim, self.args.rnd_output_dim, self.args.rnd_lr, self.args.cuda_num)
            self.rnd_buffer = []
        else:
            self.rnd = None
        if args.switch_policy:
            obs = env.reset()
            params = {'obs': obs['observation'].shape[0], 'goal': 0,
              'sub_goal': 0,
              'l_action_dim': args.l_action_dim,
              'h_action_dim': args.h_action_dim,
              'action_max': args.action_max,
              'max_timesteps': args.max_steps}
            # self.low_agent_exp = ExpLowAgent(params, args)
            # self.low_learner_exp = ExpLowLearner(self.low_agent_exp, self.monitor, args)
            self.high_agent_exp = ExpHighAgent(params, args)
            self.high_learner_exp = ExpHighLearner(self.high_agent_exp, self.monitor, args)

    def get_actions(self, ob, bg, a_max=1, random_goal=False, act_randomly=False, graph=False):
        #get subgoal
        if ((self.curr_subgoal is None) or (self.way_to_subgoal == 0)) :
            self.curr_highpolicy_obs = ob

            if random_goal:
                sub_goal = np.random.uniform(low=-1, high=1, size=self.env_params['sub_goal'])
                sub_goal = sub_goal * self.subgoal_scale + self.subgoal_offset
            
            else:
                sub_goal = self.high_agent.get_actions(ob, bg)
                if self.args.subgoal_noise_eps > 0.0:
                    subgoal_low_limit = self.subgoal_offset - self.subgoal_scale
                    subgoal_high_limit = self.subgoal_offset + self.subgoal_scale
                    sub_goal_noise = self.args.subgoal_noise_eps * np.random.randn(*sub_goal.shape)
                    sub_goal = sub_goal + sub_goal_noise
                    sub_goal = np.clip(sub_goal, subgoal_low_limit, subgoal_high_limit)

            self.curr_subgoal = sub_goal
            self.way_to_subgoal = self.subgoal_freq
            if not random_goal and self.args.dynamic_step:    
                min_rnd = self.rnd.intrinsic_reward(np.array([[0,0]])) + 0.00001
                curr_rnd = self.rnd.intrinsic_reward(np.array([ob[:self.args.subgoal_dim]]))
                scale= 0.1
                dynamic_step = int(np.clip(np.exp(-(scale*(curr_rnd/min_rnd))) * self.args.subgoal_freq, 10, self.args.subgoal_freq))
                self.monitor.store(DynamicStep=dynamic_step)
                self.monitor.store(DynamicMinRND=min_rnd)
                self.monitor.store(DynamicCurrRND=curr_rnd)

                if (dynamic_step + self.step_count_for_dynamic) > self.args.max_steps:
                    dynamic_step = (self.args.max_steps - self.step_count_for_dynamic)
                self.step_count_for_dynamic += dynamic_step
                self.way_to_subgoal = dynamic_step
                
            #graph search
            if (self.graphplanner.graph is not None):
                self.graphplanner.find_path(ob, self.curr_subgoal)

        # which waypoint to chase
        self.waypoint_subgoal = self.graphplanner.get_waypoint(ob, self.curr_subgoal)[:self.subgoal_dim]

        #find low level policy action
        if act_randomly:
            act = np.random.uniform(low=-a_max, high=a_max, size=self.env_params['l_action_dim'])
        else:
            act = self.low_agent.get_actions(ob, self.waypoint_subgoal)
            if self.args.noise_eps > 0.0:
                act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
                act = np.clip(act, -a_max, a_max)
            if self.args.random_eps > 0.0:
                a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
                mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
                if self.num_envs > 1:
                    mask = np.expand_dims(mask, -1)
                act += mask * (a_rand - act)
        self.way_to_subgoal -= 1
        return act

    
    def get_exp_action_low(self, ob, a_max=1, act_randomly=False, graph=False):
        act = self.low_agent_exp.get_actions(ob)
        if self.args.noise_eps > 0.0:
            act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
            act = np.clip(act, -a_max, a_max)
        if self.args.random_eps > 0.0:
            a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
            mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
            if self.num_envs > 1:
                mask = np.expand_dims(mask, -1)
            act += mask * (a_rand - act)
        self.way_to_subgoal -= 1
        return act


    def get_exp_action(self, ob, ag, bg, random_goal=False, act_randomly=False, a_max=1, mode='go'):
        if ((self.curr_subgoal is None) or (self.way_to_subgoal == 0)):
            self.curr_highpolicy_obs = ob
            if random_goal:
                sub_goal = np.random.uniform(low=-1, high=1, size=self.env_params['sub_goal'])
                sub_goal = sub_goal * self.subgoal_scale + self.subgoal_offset
            else:
                if mode == 'go':
                    if self.curr_subgoal is None: # sample go node once
                        sub_goal, is_reachable = self.graphplanner._get_subgoal_with_graph(ob, bg, self.rnd)
                        self.is_reachable = is_reachable
                    else:
                        sub_goal = self.curr_subgoal
                elif mode == 'exp':
                    if self.args.rule_based_sg and self.graphplanner.graph is not None:
                        sub_goal = self.graphplanner._get_rulebased_subgoal(ag.copy(), self.rnd, ob, bg)
                    else:
                        sub_goal = self.high_agent_exp.get_actions(ob) #+ ob[:self.args.subgoal_dim]
                        if self.args.subgoal_noise_eps > 0.0:
                            subgoal_low_limit = self.subgoal_offset - self.subgoal_scale
                            subgoal_high_limit = self.subgoal_offset + self.subgoal_scale
                            sub_goal_noise = self.args.subgoal_noise_eps * np.random.randn(*sub_goal.shape)
                            sub_goal = sub_goal + sub_goal_noise
                            sub_goal = np.clip(sub_goal, subgoal_low_limit, subgoal_high_limit)
                else:
                    assert False, "Wrong Mode"
            self.curr_subgoal = sub_goal
            self.way_to_subgoal = self.subgoal_freq
            # if mode == 'exp' and self.graphplanner.graph is not None: # ablation0
            #     self.way_to_subgoal = max(self.graphplanner.mean_step_edge_len, self.subgoal_freq)
                

        ##### LOW ####
        if (self.graphplanner.graph is not None) and mode == 'go':
            self.graphplanner.find_path(ob, self.curr_subgoal)
            self.waypoint_subgoal = self.graphplanner.get_waypoint(ob, self.curr_subgoal)[:self.subgoal_dim]
            # if (self.waypoint_subgoal == self.curr_subgoal).all():
            #     if(self.low_agent._get_point_to_point(ob, self.curr_subgoal) > self.args.cutoff * 2) :
            #         self.curr_subgoal = ob[:self.subgoal_dim]
            #         if 'Maze' in self.args.env_name:
            #             self.curr_subgoal = ob[:self.subgoal_dim]
            #         elif 'Reacher' in self.args.env_name:
            #             self.curr_subgoal = self.env.get_EE_pos(ob[None]).squeeze()
        else:
            self.waypoint_subgoal = self.curr_subgoal

        if act_randomly:
            act = np.random.uniform(low=-a_max, high=a_max, size=self.env_params['l_action_dim'])
        else:
            act = self.low_agent.get_actions(ob, self.waypoint_subgoal)
            if self.args.noise_eps > 0.0:
                act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
                act = np.clip(act, -a_max, a_max)
            if self.args.random_eps > 0.0:
                a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
                mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
                if self.num_envs > 1:
                    mask = np.expand_dims(mask, -1)
                act += mask * (a_rand - act)
        self.way_to_subgoal -= 1
        return act

    def rnd_optimize(self):
        batch = np.array(self.rnd_buffer)[:,None]
        loss = self.rnd.update(batch)
        self.monitor._sw.add_scalar(f'rnd/loss', loss, self.env_steps)
        self.rnd_buffer = []

    def low_agent_optimize(self):
        self.timer.start('low_train')
        
        for n_train in range(self.args.n_batches):
            batch = self.low_replay.sample(batch_size=self.args.batch_size, monitor=self.monitor, rnd=self.rnd)
            self.low_learner.update_critic(batch, train_embed=True)
            batch_g = self.low_replay.sample_g(batch_size=self.args.batch_size, monitor=self.monitor, rnd=self.rnd)
            self.low_learner.update_critic_g(batch_g, train_embed=True)

            if (self.args.use_low_intrinsic) and self.args.go_sg_with_critic:
                batch_int = self.low_replay.sample_int(batch_size=self.args.batch_size, monitor=self.monitor, rnd=self.rnd)
                self.low_learner.update_critic_int(batch_int, train_embed=True)

            if self.low_opt_steps % self.args.actor_update_freq == 0:
                self.low_learner.update_actor(batch, train_embed=True)
            self.low_opt_steps += 1
            if self.low_opt_steps % self.args.target_update_freq == 0:
                self.low_learner.target_update()
        
        self.timer.end('low_train')
        self.monitor.store(LowTimePerTrainIter=self.timer.get_time('low_train') / self.args.n_batches)


    def high_agent_optimize(self):
        self.timer.start('high_train')
        
        for n_train in range(self.args.n_batches):
            if self.args.switch_policy:
                batch_exp = self.high_replay.sample_exp(batch_size=self.args.batch_size, graphplanner=self.graphplanner, rnd=self.rnd)
                self.high_learner_exp.update_critic(batch_exp, train_embed=True)
                if self.high_opt_steps % self.args.actor_update_freq == 0:
                    self.high_learner_exp.update_actor(batch_exp, train_embed=True)
                if self.high_opt_steps % self.args.target_update_freq == 0:
                    self.high_learner_exp.target_update()

            else:
                batch = self.high_replay.sample(batch_size=self.args.batch_size, graphplanner=self.graphplanner, rnd=self.rnd)
                self.high_learner.update_critic(batch, train_embed=True)
                if self.high_opt_steps % self.args.actor_update_freq == 0:
                    self.high_learner.update_actor(batch, train_embed=True)
                self.high_opt_steps += 1
                if self.high_opt_steps % self.args.target_update_freq == 0:
                    self.high_learner.target_update()
                            
            
        
        self.timer.end('high_train')
        self.monitor.store(HighTimePerTrainIter=self.timer.get_time('high_train') / self.args.n_batches)

    
    def collect_experience(self, random_goal= False, act_randomly=False, train_agent=True, graph=False, video=False):
        low_ob_list, low_ag_list, low_bg_list, low_a_list = [], [], [], []
        high_ob_list, high_ag_list, high_bg_list, high_a_list, high_a_done_list = [], [], [], [], []
        self.monitor.update_episode()
        observation = self.env.reset()
        self.curr_subgoal = None
        ob = observation['observation']
        ag = observation['achieved_goal']
        bg = observation['desired_goal']
        ag_origin = ag.copy()
        a_max = self.env_params['action_max']
        mode = 'go'
        changed_step = 0
        
        if (self.graphplanner.graph is not None) and (self.args.FGS):
            FGS = self.graphplanner.check_easy_goal(ob, bg)
            if FGS is not None:
                bg = FGS
        a = ag.copy()
        for timestep in range(self.env_params['max_timesteps']):
            if video:            
                if "Reacher" in self.args.env_name or "Pusher" in self.args.env_name or "Bottle" in self.args.env_name:
                    if not "Pusher" in self.args.env_name:
                        frame = self.env.render(mode='rgb_array')
                        frame = frame.transpose(2, 0, 1)
                        self.env_frames.append(frame)
                else:
                    frame = self.env.base_env.render(mode='rgb_array')
                    frame = frame.transpose(2, 0, 1)
                    self.env_frames.append(frame)

            if self.args.switch_policy:
                act = self.get_exp_action(ob, ag, bg, random_goal=random_goal, act_randomly=act_randomly, a_max=a_max, mode=mode)

            else:
                act = self.get_actions(ob, bg, a_max=a_max, random_goal= random_goal, act_randomly=act_randomly, graph=graph)           
            
            low_ob_list.append(ob.copy())
            low_ag_list.append(ag.copy())
            low_bg_list.append(self.waypoint_subgoal.copy())
            low_a_list.append(act.copy())
            self.rnd_buffer.append(ag.copy())

            high_a_done_flag = False
            if ((self.way_to_subgoal == 0) or (timestep == self.env_params['max_timesteps'] - 1)):
                high_ob_list.append(self.curr_highpolicy_obs.copy())
                high_ag_list.append(self.curr_highpolicy_obs[:self.args.subgoal_dim].copy())
                high_bg_list.append(bg.copy())
                high_a_list.append(self.curr_subgoal.copy())
                high_a_done_flag = True

            high_a_done_list.append(high_a_done_flag)
            observation, _, _, info = self.env.step(act)
            ob = observation['observation']
            ag = observation['achieved_goal']
            
            if self.args.switch_policy and mode =='go' and (np.linalg.norm(ag - self.curr_subgoal) < self.args.subgoaltest_threshold) and (not self.is_reachable):
                print("mode changed: ", ag, self.curr_subgoal)
                changed_step = len(low_ob_list)
                mode='exp'

            if self.args.exp_graph and mode == 'exp':
                self.graphplanner.temp_exp_landmarks.append(ag)
                self.graphplanner.temp_exp_states.append(ob)

            self.total_timesteps += self.num_envs
            
            for every_env_step in range(self.num_envs):
                self.env_steps += 1
                if train_agent:
                    self.low_agent_optimize()

                    if (self.env_steps % self.args.high_optimize_freq == 0) and not self.args.rule_based_sg:
                        self.high_agent_optimize()
                    if self.args.rnd and (self.env_steps % self.args.rnd_batch_size == 0):
                        self.rnd_optimize()

        low_ob_list.append(ob.copy())
        low_ag_list.append(ag.copy())
        high_ob_list.append(ob.copy())
        high_ag_list.append(ag.copy())
      
        
        low_experience = dict(ob=low_ob_list, ag=low_ag_list, bg=low_bg_list, a=low_a_list)
        high_experience = dict(ob=high_ob_list, ag=high_ag_list, bg=high_bg_list, a=high_a_list)
        low_experience = {k: np.array(v) for k, v in low_experience.items()}
        high_experience = {k: np.array(v) for k, v in high_experience.items()}

        if low_experience['ob'].ndim == 2:
            low_experience = {k: np.expand_dims(v, 0) for k, v in low_experience.items()}
        else:
            low_experience = {k: np.swapaxes(v, 0, 1) for k, v in low_experience.items()}

        if high_experience['ob'].ndim == 2:
            high_experience = {k: np.expand_dims(v, 0) for k, v in high_experience.items()}
        else:
            high_experience = {k: np.swapaxes(v, 0, 1) for k, v in high_experience.items()}

        low_reward = self.low_reward_func(ag, self.waypoint_subgoal.copy(), None)
        high_reward = self.high_reward_func(ag, bg, None, ob)
        self.step_count_for_dynamic = 0
        self.monitor.store(LowReward=np.mean(low_reward))
        self.monitor.store(HighReward=np.mean(high_reward))
        self.monitor.store(Train_GoalDist=((bg - ag) ** 2).sum(axis=-1).mean())
        self.low_replay.store(low_experience)
        # self.high_replay.store(high_experience)
        self.way_to_subgoal = 0   

        return low_ob_list, high_a_list, bg, changed_step
    
    def run(self):
        for n_init_rollout in range(self.args.n_initial_rollouts // self.num_envs):
            self.collect_experience(random_goal= True, act_randomly=True, train_agent=False, graph=False)
            

        
        for epoch in range(self.args.n_epochs):
            print('Epoch %d: Iter (out of %d)=' % (epoch, self.args.n_cycles), end=' ')
            sys.stdout.flush()

            if epoch >= self.args.start_planning_epoch :
                self.graphplanner.graph_construct(epoch, self.graphplanner, self.monitor, self.env_steps)
            
            for n_iter in range(self.args.n_cycles):
                print("%d" % n_iter, end=' ' if n_iter < self.args.n_cycles - 1 else '\n')
                sys.stdout.flush()
                self.timer.start('rollout')
                video = False
                if (epoch % self.args.env_video_freq == 0) and (n_iter == 0):
                    video = True
                obs, sgs, bg, changed_step = self.collect_experience(train_agent=True, graph=True, video=video)
                if (epoch % self.args.env_video_freq == 0) and (n_iter == 0) and (not "Pusher" in self.args.env_name):
                    self.env_render(self.env_frames, self.monitor._sw, self.total_timesteps)
                # Add Exploration Nodes to current graph
                if self.args.exp_graph and (len(self.graphplanner.temp_exp_landmarks) > 0):
                    temp_l = np.array(self.graphplanner.temp_exp_landmarks)
                    temp_s = np.array(self.graphplanner.temp_exp_states)
                    self.graphplanner.exp_landmarks = np.concatenate((temp_l, self.graphplanner.exp_landmarks))
                    self.graphplanner.exp_states = np.concatenate((temp_s, self.graphplanner.exp_states))
                    novelty_sort = self.rnd.intrinsic_reward(self.graphplanner.exp_landmarks).squeeze().argsort()[::-1][:500]
                    self.graphplanner.exp_landmarks = self.graphplanner.exp_landmarks[novelty_sort]
                    self.graphplanner.exp_states = self.graphplanner.exp_states[novelty_sort]
                    self.graphplanner.temp_exp_landmarks = []
                    self.graphplanner.temp_exp_states = []

                self.timer.end('rollout')
                self.monitor.store(TimePerSeqRollout=self.timer.get_time('rollout'))
                if n_iter % 5 == 0 and self.graphplanner.landmarks is not None:
                    if self.args.use_low_intrinsic:
                        graph_dist = self.low_agent._get_dist_to_goal(self.graphplanner.states, bg[None], use_intrinsic=False)
                        self.plotting_graph_func(self.graphplanner.landmarks, bg, graph_dist, self.total_timesteps, self.monitor._sw, "graph_dist")
                        if self.args.go_sg_with_critic:
                            graph_dist_int = self.low_agent._get_dist_to_goal(self.graphplanner.states, bg[None], use_intrinsic=True)
                        else:
                            graph_dist_int = self.low_agent._get_dist_to_goal(self.graphplanner.states, bg[None])
                            if not self.args.sg_with_no_int:
                                scaled_dist = graph_dist_int.copy() * 0.3
                                graph_dist_int -= np.clip(self.rnd.intrinsic_reward(self.graphplanner.landmarks), 0., 1.0) * scaled_dist
                        
                        ##### graph plot #####
                        # graph_dist_int_rnd_only = self.rnd.intrinsic_reward(self.graphplanner.landmarks)
                        # _bg = bg[None].copy()
                        # _batch_size = _bg.shape[0]
                        # state_size = self.graphplanner.states.shape[0]
                        # goal_repeat = np.repeat(_bg[:,None], state_size, axis=1) # batch, 31, subg
                        # state_repeat = np.repeat(self.graphplanner.states[None,:], _batch_size, axis=0) # batch, 31, 29
                        # inputs = self.low_agent._process_inputs_critic(state_repeat, goal_repeat).reshape(_batch_size * state_size, self.graphplanner.states.shape[-1] + _bg.shape[-1])
                        # graph_dist_int_rnd_critic = self.low_agent.critic1_int(inputs).cpu().detach().numpy()

                        # self.plotting_graph_func(self.graphplanner.landmarks, bg, graph_dist_int_rnd_only, self.total_timesteps, self.monitor._sw, "graph_dist_int_rnd_only")    
                        # self.plotting_graph_func(self.graphplanner.landmarks, bg, graph_dist_int_rnd_critic, self.total_timesteps, self.monitor._sw, "graph_dist_int_rnd_critic")
                        self.plotting_graph_func(self.graphplanner.landmarks, bg, graph_dist_int, self.total_timesteps, self.monitor._sw, "graph_dist_int")
                    if not "Pusher"  in self.args.env_name:
                        self.plotting_func(self.graphplanner.landmarks, sgs, self.total_timesteps, self.monitor._sw, fg=bg)

                    obs = np.array(obs)

                    if "Maze" in self.args.env_name:
                        self.plotting_func(obs[:, :2], sgs, self.total_timesteps, self.monitor._sw, logpath='obs_sgs', fg=bg, changed_step=changed_step)
                    elif "Reacher" in self.args.env_name :
                        self.plotting_func(self.env.get_EE_pos(obs), sgs, self.total_timesteps, self.monitor._sw, logpath='obs_sgs', fg=bg, changed_step=changed_step)
                
            self.monitor.store(env_steps=self.env_steps)
            self.monitor.store(low_opt_steps=self.low_opt_steps)
            self.monitor.store(high_opt_steps=self.high_opt_steps)
            self.monitor.store(low_replay_size=self.low_replay.current_size)
            self.monitor.store(low_replay_fill_ratio=float(self.low_replay.current_size / self.low_replay.size))
            # self.monitor.store(high_replay_size=self.high_replay.current_size)
            # self.monitor.store(high_replay_fill_ratio=float(self.high_replay.current_size / self.high_replay.size))
            if epoch % self.args.eval_freq == 0:
                her_success = self.run_eval(epoch, use_test_env=True, render=self.args.eval_render)
                print('Epoch %d her eval %.3f'%(epoch, her_success))
                print('Log Path:', self.log_path)
                # logger.record_tabular("Epoch", epoch)
                self.monitor.store(Success_Rate=her_success)
            self.save_all(self.model_path)




    def run_eval(self, epoch, use_test_env=False, render=False):
        env = self.env
        if use_test_env and hasattr(self, 'test_env'):
            print("use test env")
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        for n_test in range(self.args.n_test_rollouts):
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            print(ag)
            for timestep in range(self.env_params['max_timesteps']):
                act = self.eval_get_actions(ob, bg)
                if render:
                    env.render()
                observation, _, _, info = env.step(act)
                ob = observation['observation']
                ag = observation['achieved_goal']
            TestEvn_Dist = env.goal_distance(ag, bg)
            print(ag,bg, TestEvn_Dist)
            self.monitor.store(TestEvn_Dist=np.mean(TestEvn_Dist))
            
            total_trial_count += 1
            if(self.args.env_name == "AntMazeSmall-v0"):
                if (TestEvn_Dist <= 2.5):
                    total_success_count += 1
            elif(self.args.env_name == "Reacher3D-v0" ):
                if (TestEvn_Dist <= 0.25):
                    total_success_count += 1
            elif self.args.env_name == "Pusher-v4":
                if (TestEvn_Dist <= 0.15):
                    total_success_count += 1
            else:
                if (TestEvn_Dist <= 5):
                    total_success_count += 1
        success_rate = total_success_count / total_trial_count
        return success_rate

    

    def eval_get_actions(self, ob, bg, a_max=1, random_goal=False, act_randomly=False, graph=False):
        if ((self.curr_subgoal is None) or (self.way_to_subgoal == 0)) :
            self.curr_highpolicy_obs = ob
            sub_goal = self.high_agent.get_actions(ob, bg)
            if self.args.switch_policy:
                sub_goal = bg
            self.curr_subgoal = sub_goal
            self.way_to_subgoal = self.subgoal_freq
            if (self.graphplanner.graph is not None):
                self.graphplanner.find_path(ob, self.curr_subgoal)

        # which waypoint to chase
        
        self.waypoint_subgoal = self.graphplanner.get_waypoint(ob, self.curr_subgoal, eval=True)
        act = self.low_agent.get_actions(ob, self.waypoint_subgoal)
        self.way_to_subgoal -= 1 
        return act


    
    def state_dict(self):
        return dict(total_timesteps=self.total_timesteps)
    
    def load_state_dict(self, state_dict):
        self.total_timesteps = state_dict['total_timesteps']
 
    def plotting_graph_func(self, nodes, fg, dist, global_step, logger, logpath):
        nodes = np.array(nodes)
        fig, ax = plt.subplots()
        if (self.args.env_name == 'AntMaze') or (self.args.env_name == 'AntMazeBottleneck-v0'):
            plt.xlim(-6,22)
            plt.ylim(-6,22)    
        elif self.args.env_name == 'AntMazeComplex-v0':
            plt.xlim(-6,55)
            plt.ylim(-6,55)
        
        plt.scatter(nodes[:,0], nodes[:,1], c=dist, alpha=0.8)
        plt.colorbar() 
        plt.scatter(fg[0], fg[1], marker='*', s=100, c='black')
        

        if eval:
            logger.add_figure(f'Eval/figures/{logpath}', plt.gcf(), global_step)
        else:
            logger.add_figure(f'figures/{logpath}', plt.gcf(), global_step)
        plt.close()

    def plotting_func(self, state_traj, sgoal_traj, global_step, logger, logpath='graph_subgoals', fg= None, changed_step=None):
        sgs = np.array(sgoal_traj)
        sgs_idx = np.arange(len(sgs))
        states = np.array(state_traj)
        st_idx =np.arange(len(states))
        max_s = len(states)
        if changed_step is None:
            changed_step = 0

        exp = max_s - changed_step
        norm = changed_step
        exp_idx =np.arange(exp)
        norm_idx =np.arange(norm)

        if "Maze" in self.args.env_name:
            

            fig, ax = plt.subplots()
            if (self.args.env_name == 'AntMaze') or (self.args.env_name == 'AntMazeBottleneck-v0'):
                plt.xlim(-6,22)
                plt.ylim(-6,22)    
            elif self.args.env_name == 'AntMazeComplex-v0':
                plt.xlim(-6,55)
                plt.ylim(-6,55)
            
            plt.scatter(states[:changed_step,0], states[:changed_step,1], c=norm_idx,cmap='winter',alpha=0.5)
            plt.colorbar() 
            plt.scatter(states[changed_step:,0], states[changed_step:,1], c=exp_idx,cmap='spring',alpha=0.5)
            plt.colorbar() 
            
            plt.scatter(sgs[:,0], sgs[:,1], c=sgs_idx, marker='x',alpha=0.8, cmap='winter')
            plt.colorbar()
            plt.scatter(fg[0], fg[1], marker='*', s=100, c='black')
        else:
            copyed_g = self.graphplanner.graph.copy()
            copyed_g.remove_edges_from(nx.selfloop_edges(copyed_g))
            plt.figure(figsize=(45,45))

            # 3d spring layout
            pos = self.graphplanner.landmarks
            # Extract node and edge positions from the layout
            node_xyz = np.array([pos[v] for v in sorted(copyed_g)])
            edge_xyz = np.array([(pos[u], pos[v]) for u, v in copyed_g.edges()])

            # Create the 3D figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Plot the nodes - alpha is scaled by "depth" automatically
            ax.scatter(*node_xyz.T, s=100, ec="w")

            # Plot the edges
            for vizedge in edge_xyz:
                ax.plot(*vizedge.T, color="tab:gray", marker='>')


            def _format_axes(ax):
                """Visualization options for the 3D axes."""
                # Turn gridlines off
                ax.grid(False)
                for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                    dim.set_ticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
                # Suppress tick labels
                
                # Set axes labels
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")

            ax.scatter(*states[:changed_step].T, c=norm_idx,cmap='winter',alpha=0.5)
            ax.scatter(*states[changed_step:].T, c=exp_idx,cmap='spring',alpha=0.5)
            ax.scatter(*sgs.T, c=sgs_idx, marker='x',alpha=0.8, cmap='winter')
            ax.scatter(*fg.T, marker='*', s=100, c='black')
            _format_axes(ax)
            fig.tight_layout()
        

        if eval:
            logger.add_figure(f'Eval/figures/{logpath}', plt.gcf(), global_step)
        else:
            logger.add_figure(f'figures/{logpath}', plt.gcf(), global_step)
        plt.close()
    

    def render(self, bound_low, bound_high, dots=5000):
        s = np.zeros([1, 29])
        s = np.repeat(s,repeats=dots, axis=0)
        sgs = np.random.uniform(bound_low, bound_high, size=([dots, bound_high.shape[0]]))
        
        return s, sgs


    def env_render(self, frames, logger, global_step):
        print(np.array(frames)[None, ...].shape)
        logger.add_video('Env_render', np.array(frames)[None, ...], global_step, fps=16)
        self.env_frames = []
        