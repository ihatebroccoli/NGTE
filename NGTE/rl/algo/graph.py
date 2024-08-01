import numpy as np
import sys
import time
import networkx as nx
import torch
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GraphPlanner:
    def __init__(self, args, low_replay, low_agent, env):
        self.low_replay = low_replay
        self.low_agent = low_agent
        self.env = env
        self.dim = args.subgoal_dim
        self.args = args

        self.graph = None
        self.n_graph_node = 0
        self.cutoff = args.cutoff
        self.landmarks = None
        self.states = None
        self.waypoint_vec = None
        self.waypoint_idx = 0
        self.waypoint_chase_step = 0
        self.waypoint_chase_step_from_last_sg = 0
        self.edge_lengths = None
        self.initial_sample = args.initial_sample
        self.mean_edge_len = None
        self.mean_step_edge_len = None
        self.temp_exp_landmarks = []
        self.temp_exp_states = []
        self.exp_states = np.zeros([1,29])
        self.exp_landmarks = np.zeros([1,self.args.subgoal_dim])

        random.seed(self.args.seed)


    def fps_selection(
            self,
            landmarks,
            states,
            n_select: int,
            inf_value=1e6,
            low_bound_epsilon=10, early_stop=True,
    ):
        n_states = landmarks.shape[0]
        dists = np.zeros(n_states) + inf_value
        chosen = []
        while len(chosen) < n_select:
            if (np.max(dists) < low_bound_epsilon) and early_stop and (len(chosen) > self.args.n_graph_node/10):
                break
            idx = np.argmax(dists)  # farthest point idx
            farthest_state = states[idx]
            chosen.append(idx)
            # distance from the chosen point to all other pts
            if self.args.use_oracle_G:
                new_dists = self._get_dist_from_start_oracle(farthest_state, landmarks)
            else:
                new_dists = self.low_agent._get_dist_from_start(farthest_state, landmarks) # 변경점
            new_dists[idx] = 0
            dists = np.minimum(dists, new_dists)
        return chosen
    
    def graph_construct(self, iter,  graphplanner, monitor, global_step, exp_graph_construct=False):
        replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
        landmarks = replay_data['ag']
        states = replay_data['ob']
        sum_edge_len = []
        step_edge_len = []
        sum_connected_edge = 0
        if len(self.exp_landmarks) > 0 and self.args.exp_graph:
            landmarks = np.concatenate((replay_data['ag'], self.exp_landmarks))
            states = np.concatenate((replay_data['ob'], self.exp_states))

        idx = self.fps_selection(
            landmarks=landmarks, 
            states=states, 
            n_select=self.args.n_graph_node,
            early_stop=True,
            low_bound_epsilon=self.args.low_bound_epsilon,
        )
        self.n_graph_node = len(idx)
        self.landmarks = landmarks[idx]
        self.states = states[idx]
        
        #get pairwise dist
        if self.args.use_oracle_G:
            pdist = self._get_pairwise_dist_oracle(self.states)
        else:
            pdist = self.low_agent._get_pairwise_dist(self.states, self.landmarks)
        self.graph = nx.DiGraph()

        # union-find
        group, node, dist = int, int, float
        shortest_dist: dict[tuple[group, group], tuple[dist, node, node]] = dict()
        parent = [i for i in range(self.n_graph_node)]
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(p, c):
            p, c = find(p), find(c)
            parent[c] = p
        # union-find end

        for i in range(self.n_graph_node):
            for j in range(self.n_graph_node):
                if i == 0:
                    min_from_i = min(pdist[i,(i+1):])
                elif i == (self.n_graph_node-1):
                    min_from_i = min(pdist[i,:i])
                else:
                    min_from_i = min(min(pdist[i,:i]), min(pdist[i,(i+1):]))
                if (pdist[i][j] < self.cutoff) or (pdist[i][j] < min_from_i * 1.2):
                    if 'Maze' in self.args.env_name:
                        sum_edge_len.append(np.linalg.norm(self.states[i][:self.args.subgoal_dim] - self.landmarks[j]))
                    elif 'Reacher' in self.args.env_name: 
                        sum_edge_len.append(np.linalg.norm(self.env.get_EE_pos(self.states[i][None]).squeeze() - self.landmarks[j]))
                    elif 'Pusher' in self.args.env_name: 
                        sum_edge_len.append(np.linalg.norm(self.landmarks[i] - self.landmarks[j]))
                    sum_connected_edge += 1
                    step_edge_len.append(pdist[i][j])
                    self.graph.add_edge(i, j, weight=pdist[i][j])
                    union(i, j)

        # Cluster 연결
        for i in range(self.n_graph_node):
            for j in range(self.n_graph_node):
                g1, g2 = find(i), find(j)
                if g1 != g2:
                    if (g1, g2) not in shortest_dist or shortest_dist[(g1, g2)][0] > pdist[i][j]:
                        shortest_dist[(g1, g2)] = (pdist[i][j], i, j)

        
        # Fully connect clusters
        for gs, cl_edge in sorted(shortest_dist.items(), key=lambda x: x[1][0]):
            self.graph.add_edge(cl_edge[1], cl_edge[2], weight=cl_edge[0])
            union(gs[0], gs[1])
        # Cluster 연결 끝
        
        self.mean_edge_len = np.mean(sum_edge_len)
        self.mean_step_edge_len = np.mean(step_edge_len)
        print("subgoal_mean_step_edge_len : ", self.mean_step_edge_len)
        
        monitor.store(subgoal_mean_step_edge_len=self.mean_step_edge_len)
        #######Graph print#######
        if 'Maze' in self.args.env_name:
            copyed_g = self.graph.copy()
            copyed_g.remove_edges_from(nx.selfloop_edges(copyed_g))
            plt.figure(figsize=(30,30))
            pos = np.concatenate((self.landmarks, self.states[:,:2]))
            nx.draw_networkx(copyed_g,pos=pos,node_size=45,font_size=6)
            monitor._sw.add_figure(f'graph/graph_fig', plt.gcf(), global_step)
            plt.close()

        elif 'Reacher' in self.args.env_name:
            copyed_g = self.graph.copy()
            copyed_g.remove_edges_from(nx.selfloop_edges(copyed_g))
            plt.figure(figsize=(45,45))

            # 3d spring layout
            pos = self.landmarks
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
                ax.plot(*vizedge.T, color="tab:gray")


            def _format_axes(ax):
                """Visualization options for the 3D axes."""
                # Turn gridlines off
                ax.grid(False)
                # Suppress tick labels
                for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                    dim.set_ticks(np.arange(-1,1,0.25))
                # Set axes labels
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")


            _format_axes(ax)
            fig.tight_layout()
            monitor._sw.add_figure(f'graph/graph_fig', plt.gcf(), global_step)
            plt.close()

        # elif 'Pusher' in self.args.env_name:
        #     copyed_g = self.graph.copy()
        #     copyed_g.remove_edges_from(nx.selfloop_edges(copyed_g))
        #     plt.figure(figsize=(45,45))

        #     # 3d spring layout
        #     pos = self.landmarks
        #     # Extract node and edge positions from the layout
        #     node_xyz = np.array([pos[v] for v in sorted(copyed_g)])
        #     edge_xyz = np.array([(pos[u], pos[v]) for u, v in copyed_g.edges()])

        #     # Create the 3D figure
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection="3d")

        #     # Plot the nodes - alpha is scaled by "depth" automatically
        #     ax.scatter(*node_xyz.T, s=100, ec="w")

        #     # Plot the edges
        #     for vizedge in edge_xyz:
        #         ax.plot(*vizedge.T, color="tab:gray")


        #     def _format_axes(ax):
        #         """Visualization options for the 3D axes."""
        #         # Turn gridlines off
        #         ax.grid(False)
        #         # Suppress tick labels
        #         for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        #             dim.set_ticks(np.arange(-1,1,0.25))
        #         # Set axes labels
        #         ax.set_xlabel("x")
        #         ax.set_ylabel("y")
        #         ax.set_zlabel("z")


        #     _format_axes(ax)
        #     fig.tight_layout()
        #     monitor._sw.add_figure(f'graph/graph_fig', plt.gcf(), global_step)
        #     plt.close()

        return self.landmarks, self.states
    
    def dist_from_graph_to_goal(self, subgoal):
        dist_list=[]
        for i in range(subgoal.shape[0]):  
            curr_subgoal = subgoal[i,:self.dim]
            if self.args.use_oracle_G:
                goal_edge_length = self._get_dist_to_goal_oracle(self.states, curr_subgoal)
            else:
                goal_edge_length = self.low_agent._get_dist_to_goal(self.states, curr_subgoal)
            dist_list.append(min(goal_edge_length))

        return np.array(dist_list)  

    def sg_find_path(self, ob, sg_idx, inf_value=1e6):
        expanded_graph = self.graph.copy()
        subgoal = self.landmarks[sg_idx].copy()

        start_edge_length = self.low_agent._get_dist_from_start(ob, self.landmarks)
        for i in range(self.n_graph_node):
            if(start_edge_length[i] < self.cutoff):
                expanded_graph.add_edge('start', i, weight = start_edge_length[i])
        path = nx.shortest_path(expanded_graph, 'start', 'goal', weight='weight')
        

    def find_path(self, ob, subgoal, inf_value=1e6):
        expanded_graph = self.graph.copy()
        subgoal = subgoal[:self.dim]
        if self.args.use_oracle_G:
            start_edge_length = self._get_dist_from_start_oracle(ob, self.landmarks)
        else:
            start_edge_length = self.low_agent._get_dist_from_start(ob, self.landmarks) #변경점
        if self.args.use_oracle_G:
            goal_edge_length = self._get_dist_to_goal_oracle(self.states, subgoal)
        else:
            goal_edge_length = self.low_agent._get_dist_to_goal(self.states, subgoal[None]) #변경점
        for i in range(self.n_graph_node):
            if(start_edge_length[i] < self.cutoff):
                expanded_graph.add_edge('start', i, weight = start_edge_length[i])
            if(goal_edge_length[i] < self.cutoff):
                expanded_graph.add_edge(i, 'goal', weight = goal_edge_length[i])

        if self.args.use_oracle_G:
            start_to_goal_length = np.squeeze(self._get_point_to_point_oracle(ob, subgoal))
        else:
            start_to_goal_length = np.squeeze(self.low_agent._get_point_to_point(ob, subgoal)) #변경점
        if start_to_goal_length < self.cutoff:
            expanded_graph.add_edge('start', 'goal', weight = start_to_goal_length)
        
        self.edge_lengths = [] 
        if((not expanded_graph.has_node('start')) or (not expanded_graph.has_node('goal')) or (not nx.has_path(expanded_graph, 'start', 'goal'))):
            #if no edge from start point, force to make edge from start point
            if(not expanded_graph.has_node('start')):
                adjusted_cutoff = min(start_edge_length) * 1.5
                for i in range(self.n_graph_node):
                    if(start_edge_length[i] < adjusted_cutoff):
                        expanded_graph.add_edge('start', i, weight = start_edge_length[i])
            #check whether new path has made or not
            if(expanded_graph.has_node('goal')) and (nx.has_path(expanded_graph, 'start', 'goal')):
                path = nx.shortest_path(expanded_graph, 'start', 'goal', weight='weight')
                for (i, j) in zip(path[:-1], path[1:]):
                    self.edge_lengths.append(expanded_graph[i][j]['weight'])
            #if don't have path even though we made edges from start point
            else:
                while True:
                    nearestnode = np.argmin(goal_edge_length) #nearest point from the goal
                    if(expanded_graph.has_node(nearestnode)) and (nx.has_path(expanded_graph, 'start', nearestnode)):
                        path = nx.shortest_path(expanded_graph, 'start', nearestnode, weight='weight')
                        for (i, j) in zip(path[:-1], path[1:]):
                            self.edge_lengths.append(expanded_graph[i][j]['weight'])
                        path.append('goal')
                        self.edge_lengths.append(min(goal_edge_length))
                        break
                    else:
                        goal_edge_length[nearestnode] = inf_value #if that nearst point don't have path from start, remove it.
        elif(nx.has_path(expanded_graph, 'start', 'goal')):
            path = nx.shortest_path(expanded_graph, 'start', 'goal', weight='weight')
            for (i, j) in zip(path[:-1], path[1:]):
                self.edge_lengths.append(expanded_graph[i][j]['weight'])
        
        self.waypoint_vec = list(path)[1:-1]
        self.waypoint_idx = 0
        self.waypoint_chase_step = 0

    def check_easy_goal(self, ob, subgoal):
        expanded_graph = self.graph.copy()
        subgoal = subgoal[:self.dim]
        
        if self.args.use_oracle_G:
            start_edge_length = self._get_dist_from_start_oracle(ob, self.landmarks)
        else:
            start_edge_length = self.low_agent._get_dist_from_start(ob, self.landmarks)
        if self.args.use_oracle_G:
            goal_edge_length = self._get_dist_to_goal_oracle(self.states, subgoal)

        else:
            goal_edge_length = self.low_agent._get_dist_to_goal(self.states, subgoal[None])
            
        for i in range(self.n_graph_node):
            if(start_edge_length[i] < self.cutoff):
                expanded_graph.add_edge('start', i, weight = start_edge_length[i])
            if(goal_edge_length[i] < self.cutoff):
                expanded_graph.add_edge(i, 'goal', weight = goal_edge_length[i])

        if self.args.use_oracle_G:
            start_to_goal_length = np.squeeze(self._get_point_to_point_oracle(ob, subgoal))
        else:
            start_to_goal_length = np.squeeze(self.low_agent._get_point_to_point(ob, subgoal))
        if start_to_goal_length < self.cutoff:
            expanded_graph.add_edge('start', 'goal', weight = start_to_goal_length)
        
        if((not expanded_graph.has_node('start')) or (not expanded_graph.has_node('goal')) or (not nx.has_path(expanded_graph, 'start', 'goal'))):
            return None
        elif(nx.has_path(expanded_graph, 'start', 'goal')):
            start_edge_length = []
            for i in range (self.n_graph_node):
                if expanded_graph.has_node(i) and nx.has_path(expanded_graph, 'start', i):
                    start_edge_length.append(nx.shortest_path_length(expanded_graph, source='start', target=i, weight='weight'))
                else:
                    start_edge_length.append(5e3)
            start_edge_length = np.array(start_edge_length)
            if (start_edge_length  == 0.).all():
                farthest = 0
            else: farthest = random.choices(range(len(start_edge_length)), weights=start_edge_length)[0]
            #farthest = np.argmax(start_edge_length)
            
            return self.landmarks[farthest,:self.dim] + np.random.uniform(low=-3, high=3, size=self.args.subgoal_dim)

    
    def get_waypoint(self, ob,  subgoal, eval=False):
        if self.graph is not None:
            self.waypoint_chase_step += 1 # how long does agent chased current waypoint
            if(self.waypoint_idx >= len(self.waypoint_vec)):
                waypoint_subgoal = subgoal
            else:
                # next waypoint or not
                if((self.waypoint_chase_step > self.edge_lengths[self.waypoint_idx]) or (np.linalg.norm(ob[:self.dim]-self.landmarks[self.waypoint_vec[self.waypoint_idx]][:self.dim]) < 0.5)):
                    self.waypoint_idx += 1
                    self.waypoint_chase_step = 0

                if(self.waypoint_idx >= len(self.waypoint_vec)):
                    waypoint_subgoal = subgoal
                    self.waypoint_chase_step_from_last_sg += 1
                    if eval==False and self.waypoint_chase_step_from_last_sg > self.args.cutoff * 2:#(self.waypoint_chase_step_from_last_sg > self.low_agent._get_point_to_point(ob, subgoal) * 2):
                        if 'Maze' in self.args.env_name:
                            waypoint_subgoal = ob[:self.args.subgoal_dim].copy()
                        
                        elif 'Reacher' in self.args.env_name:
                            waypoint_subgoal = self.env.get_EE_pos(ob[None]).squeeze().copy()

                        elif 'Pusher' in self.args.env_name:
                            waypoint_subgoal = ob[[14,15,16,20,21,22]].copy()
                        else:
                            None

                        self.waypoint_chase_step_from_last_sg = 0
                else:
                    waypoint_subgoal = self.landmarks[self.waypoint_vec[self.waypoint_idx]][:self.dim]
        else:
            waypoint_subgoal = subgoal
        return waypoint_subgoal
    
    def is_goal_reachable(self, goal):
        goal_edge_length = self.low_agent._get_dist_to_goal(self.states, goal[None])
        print(goal_edge_length.min(), goal_edge_length.max())
        nearestnode_len = np.min(goal_edge_length)
        is_reachable = (nearestnode_len <= self.cutoff)
        print(is_reachable, nearestnode_len)
        return is_reachable, None

    def _get_subgoal_with_graph(self, ob, bg, rnd):
        is_reachable, _ = self.is_goal_reachable(bg)
        
        if is_reachable:
            return bg, is_reachable
        else:
            if self.args.go_sg_with_critic:
                goal_edge_length = self.low_agent._get_dist_to_goal(self.states, bg[None], use_intrinsic=True) # + self.low_agent._get_dist_from_start(ob, self.states, use_intrinsic=True)

            else: 
                goal_edge_length = self.low_agent._get_dist_to_goal(self.states, bg[None]) # 1 31 1
                if not self.args.sg_with_no_int:
                    scaled_dist = goal_edge_length.copy() * 0.5
                    goal_edge_length -= np.clip(rnd.intrinsic_reward(self.states[:, :self.args.subgoal_dim]), 0., 1.0) * scaled_dist
            
            idx = np.argmin(goal_edge_length)

            return self.landmarks[idx], is_reachable

    def _get_rulebased_subgoal(self, curr_ag, rnd, ob, bg):
        """
        use copy to curr_ag
        """
        scale = 2.
        if 'Maze' in self.args.env_name:
            # ablation0
            if np.random.uniform() < 0.5:
                sg = curr_ag + np.random.uniform(low=-self.mean_edge_len * scale, high=self.mean_edge_len * scale, size=curr_ag.shape)
            else:
                curr_ag = np.repeat(curr_ag[None, :], self.args.num_candidate_nodes ,axis=0)
                candidate_nodes =  curr_ag + np.random.uniform(low=-self.mean_edge_len * 2, high=self.mean_edge_len * 2, size=curr_ag.shape)
                sg = candidate_nodes[rnd.intrinsic_reward(candidate_nodes, update=False).argmax()]    
        
        elif 'Reacher' in self.args.env_name:
            if np.random.uniform() < 0.5:
                sg = ob
                sg = self.env.get_EE_pos(sg[None]).squeeze() + np.random.uniform(low=-self.mean_edge_len * scale, high=self.mean_edge_len * scale, size=3)
            else:
                candidate_nodes = np.repeat(ob[None, :], self.args.num_candidate_nodes ,axis=0)
                candidate_nodes = self.env.get_EE_pos(candidate_nodes) + np.random.uniform(low=-self.mean_edge_len * scale, high=self.mean_edge_len * scale, size=[self.args.num_candidate_nodes, 3])
                sg = candidate_nodes[rnd.intrinsic_reward(candidate_nodes, update=False).argmax()]
        elif 'Pusher' in self.args.env_name:
            if np.random.uniform() < 0.5:
                curr_ag += np.random.uniform(low=-self.mean_edge_len * scale, high=self.mean_edge_len * scale, size=3)
            else:
                candidate_nodes = np.repeat(curr_ag[None, :], self.args.num_candidate_nodes ,axis=0)
                candidate_nodes = candidate_nodes + np.random.uniform(low=-self.mean_edge_len * scale, high=self.mean_edge_len * scale, size=[self.args.num_candidate_nodes, 3])
                sg = candidate_nodes[rnd.intrinsic_reward(candidate_nodes, update=False).argmax()]
        else:
            print('-')
        
        return sg


    #####################oracle graph#########################
    def _get_dist_to_goal_oracle(self, obs_tensor, goal):
        goal_repeat = np.ones_like(obs_tensor[:, :self.args.subgoal_dim]) \
            * np.expand_dims(goal[:self.args.subgoal_dim], axis=0)
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - goal_repeat, axis=1)
        return dist

    def _get_dist_from_start_oracle(self, start, obs_tensor):
        start_repeat = np.ones_like(obs_tensor) * np.expand_dims(start, axis=0)
        start_repeat = start_repeat[:, :self.args.subgoal_dim]
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - start_repeat, axis=1)
        return dist

    def _get_point_to_point_oracle(self, point1, point2):
        point1 = point1[:self.args.subgoal_dim]
        point2 = point2[:self.args.subgoal_dim]
        dist = np.linalg.norm(point1-point2)
        return dist

    def _get_pairwise_dist_oracle(self, obs_tensor):
        goal_tensor = obs_tensor
        dist_matrix = []
        for obs_index in range(obs_tensor.shape[0]):
            obs = obs_tensor[obs_index]
            obs_repeat_tensor = np.ones_like(goal_tensor) * np.expand_dims(obs, axis=0)
            dist = np.linalg.norm(obs_repeat_tensor[:, :self.args.subgoal_dim] - goal_tensor[:, :self.args.subgoal_dim], axis=1)
            dist_matrix.append(np.squeeze(dist))
        pairwise_dist = np.array(dist_matrix) #pairwise_dist[i][j] is dist from i to j
        return pairwise_dist