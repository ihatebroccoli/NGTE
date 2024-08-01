GPU=$1
SEED=$2
SAVE_DIR=$3


python NGTE/main.py \
--env_name 'Pusher-v4' \
--test_env_name 'Pusher-v4' \
--action_max 20. \
--max_steps 100 \
--high_future_step 10 \
--subgoal_freq 5 \
--subgoal_scale 1. 1. 1. \
--subgoal_offset 0. 0. 0. \
--low_future_step 100 \
--subgoaltest_threshold 0.1 \
--subgoal_dim 3 \
--l_action_dim 7 \
--h_action_dim 3 \
--cutoff 10 \
--n_initial_rollouts 200 \
--n_graph_node 300 \
--low_bound_epsilon 5 \
--gradual_pen 5.0 \
--subgoal_noise_eps 0.1 \
--cuda_num ${GPU} \
--seed ${SEED} \
--save_dir ${SAVE_DIR} \
--absolute_goal \
--start_planning_epoch 0 \
--rnd \
--min_subgoal_freq 5 \
--low_int_method 'rnd' \
--use_low_intrinsic \
--switch_policy \
--int_dist_scale 0.9 \
--int_scale 1. \
--g_critic_dist_clip -99.9 \
--rule_based_sg \
--go_sg_with_critic \
--n_cycles 10 \