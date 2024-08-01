import torch
from torch.optim import Adam

import os.path as osp
from rl.utils import net_utils
import numpy as np
from rl.learn.core import dict_to_numpy


class HighLearner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        name='learner',
    ):
        self.agent = agent
        self.monitor = monitor
        self.args = args
        
        self.q_optim = Adam(list(agent.critic1.parameters())+list(agent.critic2.parameters()), lr=args.lr_critic)
        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor)
        
        self._save_file = str(name) + '.pt'
    
    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, bg, pi_target=True) + noise
            q_next = self.agent.get_qs(o2, bg, n_a, q_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)
        
        q_bg1 = self.agent.get_qs(o, bg, a, net = 1)
        q_bg2 = self.agent.get_qs(o, bg, a, net = 2)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()
        

    
        loss_critic = {'critic_1' : loss_q1, 'critic_2' : loss_q2}
        
        self.monitor.store(
            HighLoss_q1=loss_q1.item(),
            HighLoss_q2=loss_q2.item(),
            HighLoss_critic_1=loss_critic['critic_1'].item(),
            HighLoss_critic_2=loss_critic['critic_2'].item(),
        )
        monitor_log = dict(
            Highq_targ=q_targ,
            Highoffset=offset,
            Highr=r,
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_critic
    
    def actor_loss(self, batch):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']
        

        a = self.agent.to_tensor(a)
        
        q_pi, pi = self.agent.forward1(o, bg)
        subgoal_scale = torch.as_tensor(self.args.subgoal_scale, dtype=torch.float32).cuda(device=self.args.cuda_num)
        subgoal_offset = torch.as_tensor(self.args.subgoal_offset, dtype=torch.float32).cuda(device=self.args.cuda_num)
        action_l2 = ((pi - subgoal_offset) / subgoal_scale).pow(2).mean()
        loss_actor = (- q_pi).mean()
        
        pi_future = self.agent.get_pis(o, future_ag)
        loss_bc = (pi_future - a).pow(2).mean()
        
        self.monitor.store(
            HighLoss_actor=loss_actor.item(),
            HighLoss_action_l2=action_l2.item(),
            HighLoss_bc=loss_bc.item(),
        )
        monitor_log = dict(Highq_pi=q_pi)
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_actor
    
    
    def update_critic(self, batch, train_embed=True):
        loss_critic1 = self.critic_loss(batch)['critic_1']
        loss_critic2 = self.critic_loss(batch)['critic_2']
        self.q_optim.zero_grad()
        (loss_critic1+loss_critic2).backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Highgradnorm_critic1=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Highgradnorm_critic2=c_norm2)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Highgradnorm_mean_critic1=net_utils.mean_grad_norm(self.agent.critic1.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1.parameters(), self.args.grad_value_clipping)
            self.monitor.store(Highgradnorm_mean_critic2=net_utils.mean_grad_norm(self.agent.critic2.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2.parameters(), self.args.grad_value_clipping)
        self.q_optim.step()
            
    def update_actor(self, batch, train_embed=True):
        loss_actor = self.actor_loss(batch)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        
        if self.args.grad_norm_clipping > 0.:
            a_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Highgradnorm_actor=a_norm)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Highgradnorm_mean_actor=net_utils.mean_grad_norm(self.agent.actor.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.args.grad_value_clipping)
        self.pi_optim.step()
    
    def target_update(self):
        self.agent.target_update()
    
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def state_dict(self):
        return dict(
            q_optim=self.q_optim.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim.load_state_dict(state_dict['q_optim'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
    
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


class LowLearner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        name='learner',
    ):
        self.agent = agent
        self.monitor = monitor
        self.args = args
        
        self.q_optim = Adam(list(agent.critic1.parameters())+list(agent.critic2.parameters()), lr=args.lr_critic)
        self.q_optim_g = Adam(list(agent.critic1_g.parameters())+list(agent.critic2_g.parameters()), lr=args.lr_critic)
        self.q_optim_int = Adam(list(agent.critic1_int.parameters())+list(agent.critic2_int.parameters()), lr=args.lr_critic_int)

        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor)
        # self.naive_optim = Adam(agent.naive_net.parameters(), lr=args.lr_critic)
        
        self._save_file = str(name) + '.pt'
    
    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05 
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, bg, pi_target=True) + noise
            q_next = self.agent.get_qs(o2, bg, n_a, q_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)
        
        q_bg1 = self.agent.get_qs(o, bg, a, net = 1)
        q_bg2 = self.agent.get_qs(o, bg, a, net = 2)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()
        
    
        loss_critic = {'critic_1' : loss_q1, 'critic_2' : loss_q2}
        
        self.monitor.store(
            LowLoss_q1=loss_q1.item(),
            LowLoss_q2=loss_q2.item(),
            LowLoss_critic_1=loss_critic['critic_1'].item(),
            LowLoss_critic_2=loss_critic['critic_1'].item(),
        )
        monitor_log = dict(
            Lowq_targ=q_targ,
            Lowoffset=offset,
            Lowr=r,
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_critic

    def critic_loss_g(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())

        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05 
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, bg, pi_target=True) + noise
            q_next = self.agent.get_qs_g(o2, bg, n_a, q_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, max = 0.0)

        q_bg1 = self.agent.get_qs_g(o, bg, a, net = 1)
        q_bg2 = self.agent.get_qs_g(o, bg, a, net = 2)
        q_bg = self.agent.get_qs_g(o, bg, a)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()

        loss_critic = {'critic_1_g' : loss_q1, 'critic_2_g' : loss_q2}

        self.monitor.store(
            LowLoss_q1_g=loss_q1.item(),
            LowLoss_q2_g=loss_q2.item(),
            LowLoss_critic_1_g=loss_critic['critic_1_g'].item(),
            LowLoss_critic_2_g=loss_critic['critic_2_g'].item(),
        )
        return loss_critic

    def critic_loss_int(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())

        with torch.no_grad():
            q_next = self.agent.get_qs_int(o2, bg, q_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, min = 0.0)

        q_bg1 = self.agent.get_qs_int(o, bg,net = 1)
        q_bg2 = self.agent.get_qs_int(o, bg,net = 2)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()

        loss_critic = {'critic_1_int' : loss_q1, 'critic_2_int' : loss_q2}
        self.monitor.store(
            LowLoss_q1_int=loss_q1.item(),
            LowLoss_q2_int=loss_q2.item(),
            LowLoss_critic_1_int=loss_critic['critic_1_int'].item(),
            LowLoss_critic_2_int=loss_critic['critic_2_int'].item(),
        )

        return loss_critic
    
    def actor_loss(self, batch):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']
        
        a = self.agent.to_tensor(a)
        
        q_pi, pi = self.agent.forward1(o, bg)
        action_l2 = (pi / self.agent.actor.act_limit).pow(2).mean()
        loss_actor = (- q_pi).mean() + self.args.action_l2 * action_l2
        
        pi_future = self.agent.get_pis(o, future_ag)
        loss_bc = (pi_future - a).pow(2).mean()
        
        self.monitor.store(
            LowLoss_actor=loss_actor.item(),
            LowLoss_action_l2=action_l2.item(),
            LowLoss_bc=loss_bc.item(),
        )
        monitor_log = dict(Lowq_pi=q_pi)
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_actor
    

    def update_critic(self, batch, train_embed=True):
        loss_critic = self.critic_loss(batch)
        loss_critic1 = loss_critic['critic_1']
        loss_critic2 = loss_critic['critic_2']
        self.q_optim.zero_grad()
        (loss_critic1+loss_critic2).backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_critic1=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_critic2=c_norm2)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Lowgradnorm_mean_critic1=net_utils.mean_grad_norm(self.agent.critic1.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1.parameters(), self.args.grad_value_clipping)
            self.monitor.store(Lowgradnorm_mean_critic2=net_utils.mean_grad_norm(self.agent.critic2.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2.parameters(), self.args.grad_value_clipping)
        self.q_optim.step()


    def update_naive(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        # self.naive o, ag
        mseLoss = torch.nn.MSELoss()
        loss_naive = mseLoss(self.agent.naive_net(o, ag))
        self.naive_optim.zero_grad()
        loss_naive.backward()

    def update_critic_g(self, batch, train_embed=True):
        loss_critic = self.critic_loss_g(batch)
        loss_critic1 = loss_critic['critic_1_g']
        loss_critic2 = loss_critic['critic_2_g']
        self.q_optim_g.zero_grad()
        (loss_critic1+loss_critic2).backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1_g.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_critic1_g=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2_g.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_critic2_g=c_norm2)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Lowgradnorm_mean_critic1_g=net_utils.mean_grad_norm(self.agent.critic1_g.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1_g.parameters(), self.args.grad_value_clipping)
            self.monitor.store(Lowgradnorm_mean_critic2_g=net_utils.mean_grad_norm(self.agent.critic2_g.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2_g.parameters(), self.args.grad_value_clipping)
        self.q_optim_g.step()

    def update_critic_int(self, batch, train_embed=True):
        loss_critic = self.critic_loss_int(batch)
        loss_critic1 = loss_critic['critic_1_int']
        loss_critic2 = loss_critic['critic_2_int']
        self.q_optim_int.zero_grad()
        (loss_critic1+loss_critic2).backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1_int.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_critic1_int=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2_int.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_critic2_int=c_norm2)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Lowgradnorm_mean_critic1_int=net_utils.mean_grad_norm(self.agent.critic1_int.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1_int.parameters(), self.args.grad_value_clipping)
            self.monitor.store(Lowgradnorm_mean_critic2_int=net_utils.mean_grad_norm(self.agent.critic2_int.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2_int.parameters(), self.args.grad_value_clipping)
        self.q_optim_int.step()
            
    def update_actor(self, batch, train_embed=True):
        loss_actor = self.actor_loss(batch)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        
        if self.args.grad_norm_clipping > 0.:
            a_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_actor=a_norm)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Lowgradnorm_mean_actor=net_utils.mean_grad_norm(self.agent.actor.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.args.grad_value_clipping)
        self.pi_optim.step()
    
    def target_update(self):
        self.agent.target_update()
    
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def state_dict(self):
        return dict(
            q_optim=self.q_optim.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim.load_state_dict(state_dict['q_optim'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
    
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


class ExpLowLearner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        name='learner',
    ):
        self.agent = agent
        self.monitor = monitor
        self.args = args
        
        self.q_optim = Adam(list(agent.critic1.parameters())+list(agent.critic2.parameters()), lr=args.lr_critic)
        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor)
        # self.naive_optim = Adam(agent.naive_net.parameters(), lr=args.lr_critic)
        self._save_file = str(name) + '.pt'
    
    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05 
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, pi_target=True) + noise
            q_next = self.agent.get_qs(o2, n_a, q_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)
        
        q_1 = self.agent.get_qs(o, a, net = 1)
        q_2 = self.agent.get_qs(o, a, net = 2)
        loss_q1 = (q_1 - q_targ).pow(2).mean()
        loss_q2 = (q_2 - q_targ).pow(2).mean()
        
    
        loss_critic = {'critic_1' : loss_q1, 'critic_2' : loss_q2}
        
        self.monitor.store(
            ExpLowLoss_q1=loss_q1.item(),
            ExpLowLoss_q2=loss_q2.item(),
            ExpLowLoss_critic_1=loss_critic['critic_1'].item(),
            ExpLowLoss_critic_2=loss_critic['critic_2'].item(),
        )
        monitor_log = dict(
            ExpLowq_targ=q_targ,
            ExpLowoffset=offset,
            ExpLowr=r,
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_critic

    def actor_loss(self, batch):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']
        
        a = self.agent.to_tensor(a)
        
        q_pi, pi = self.agent.forward1(o)
        action_l2 = (pi / self.agent.actor.act_limit).pow(2).mean()
        loss_actor = (- q_pi).mean() + self.args.action_l2 * action_l2
        
        pi_future = self.agent.get_pis(o)
        loss_bc = (pi_future - a).pow(2).mean()
        
        self.monitor.store(
            ExpLowLoss_actor=loss_actor.item(),
            ExpLowLoss_action_l2=action_l2.item(),
            ExpLowLoss_bc=loss_bc.item(),
        )
        monitor_log = dict(ExpLowq_pi=q_pi)
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_actor
    

    def update_critic(self, batch, train_embed=True):
        loss_critic = self.critic_loss(batch)
        loss_critic1 = loss_critic['critic_1']
        loss_critic2 = loss_critic['critic_2']
        self.q_optim.zero_grad()
        (loss_critic1+loss_critic2).backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_critic1=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_critic2=c_norm2)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Lowgradnorm_mean_critic1=net_utils.mean_grad_norm(self.agent.critic1.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1.parameters(), self.args.grad_value_clipping)
            self.monitor.store(Lowgradnorm_mean_critic2=net_utils.mean_grad_norm(self.agent.critic2.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2.parameters(), self.args.grad_value_clipping)
        self.q_optim.step()

    def update_actor(self, batch, train_embed=True):
        loss_actor = self.actor_loss(batch)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        
        if self.args.grad_norm_clipping > 0.:
            a_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Lowgradnorm_actor=a_norm)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Lowgradnorm_mean_actor=net_utils.mean_grad_norm(self.agent.actor.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.args.grad_value_clipping)
        self.pi_optim.step()
    
    def target_update(self):
        self.agent.target_update()
    
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def state_dict(self):
        return dict(
            q_optim=self.q_optim.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim.load_state_dict(state_dict['q_optim'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
    
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



class ExpHighLearner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        name='learner',
    ):
        self.agent = agent
        self.monitor = monitor
        self.args = args
        
        self.q_optim = Adam(list(agent.critic1.parameters())+list(agent.critic2.parameters()), lr=args.lr_critic)
        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor)
        # self.naive_optim = Adam(agent.naive_net.parameters(), lr=args.lr_critic)
        self._save_file = str(name) + '.pt'
    
    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            # noise = np.random.randn(*a.shape) * 0.05 
            # noise = self.agent.to_tensor(noise)
            # n_a = self.agent.get_pis(o2, pi_target=True) + noise
            # q_next = self.agent.get_qs(o2, n_a, q_target=True)
            q_targ = r #+ self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)
        
        q_1 = self.agent.get_qs(o, a, net = 1)
        q_2 = self.agent.get_qs(o, a, net = 2)
        loss_q1 = (q_1 - q_targ).pow(2).mean()
        loss_q2 = (q_2 - q_targ).pow(2).mean()
        
    
        loss_critic = {'critic_1' : loss_q1, 'critic_2' : loss_q2}
        
        self.monitor.store(
            ExpHighLoss_q1=loss_q1.item(),
            ExpHighLoss_q2=loss_q2.item(),
            ExpHighLoss_critic_1=loss_critic['critic_1'].item(),
            ExpHighLoss_critic_2=loss_critic['critic_2'].item(),
        )
        monitor_log = dict(
            ExpHighq_targ=q_targ,
            ExpHighoffset=offset,
            ExpHighr=r,
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_critic

    def actor_loss(self, batch):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']
        
        a = self.agent.to_tensor(a)
        
        q_pi, pi = self.agent.forward1(o)
        action_l2 = (pi / self.agent.actor.act_limit).pow(2).mean()
        loss_actor = (- q_pi).mean() + self.args.action_l2 * action_l2
        
        pi_future = self.agent.get_pis(o)
        loss_bc = (pi_future - a).pow(2).mean()
        
        self.monitor.store(
            ExpHighLoss_actor=loss_actor.item(),
            ExpHighLoss_action_l2=action_l2.item(),
            ExpHighLoss_bc=loss_bc.item(),
        )
        monitor_log = dict(ExpLowq_pi=q_pi)
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_actor
    

    def update_critic(self, batch, train_embed=True):
        loss_critic = self.critic_loss(batch)
        loss_critic1 = loss_critic['critic_1']
        loss_critic2 = loss_critic['critic_2']
        self.q_optim.zero_grad()
        (loss_critic1+loss_critic2).backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Highgradnorm_critic1=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Highgradnorm_critic2=c_norm2)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Highgradnorm_mean_critic1=net_utils.mean_grad_norm(self.agent.critic1.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1.parameters(), self.args.grad_value_clipping)
            self.monitor.store(Highgradnorm_mean_critic2=net_utils.mean_grad_norm(self.agent.critic2.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2.parameters(), self.args.grad_value_clipping)
        self.q_optim.step()

    def update_actor(self, batch, train_embed=True):
        loss_actor = self.actor_loss(batch)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        
        if self.args.grad_norm_clipping > 0.:
            a_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(Highgradnorm_actor=a_norm)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(Highgradnorm_mean_actor=net_utils.mean_grad_norm(self.agent.actor.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.args.grad_value_clipping)
        self.pi_optim.step()
    
    def target_update(self):
        self.agent.target_update()
    
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def state_dict(self):
        return dict(
            q_optim=self.q_optim.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim.load_state_dict(state_dict['q_optim'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
    
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

