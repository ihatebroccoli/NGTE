GPU=$1
SEED=$2
SAVE_DIR=$3


python NGTE/main.py \
--env_name 'AntMazeBottleneck-v0' \
--test_env_name 'AntMazeBottleneck-eval-v0' \
--action_max 30. \
--max_steps 600 \
--high_future_step 12 \
--subgoal_freq 10 \
--subgoal_scale 12. 12. \
--subgoal_offset 8. 8. \
--low_future_step 150 \
--subgoaltest_threshold 1 \
--subgoal_dim 2 \
--l_action_dim 8 \
--h_action_dim 2 \
--cutoff 50 \
--n_initial_rollouts 10 \
--n_graph_node 300 \
--low_bound_epsilon 10 \
--gradual_pen 1.5 \
--subgoal_noise_eps 2 \
--cuda_num ${GPU} \
--seed ${SEED} \
--save_dir ${SAVE_DIR} \
--start_planning_epoch 0 \
--rnd \
--min_subgoal_freq 10 \
--low_int_method 'rnd' \
--use_low_intrinsic \
--switch_policy \
--int_dist_scale 0.9 \
--int_scale 1. \
--g_critic_dist_clip -99.9 \
--rule_based_sg \
--go_sg_with_critic \
--FGS \