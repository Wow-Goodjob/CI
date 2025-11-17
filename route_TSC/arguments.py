import argparse


def get_common_args(EXEC=False):
    ap = argparse.ArgumentParser()

    ''' curiosity DQN'''
    ap.add_argument("--forward_scale", type=float, default=0.8)  # 前向预测模型损失函数的比例为 0.8
    ap.add_argument("--inverse_scale", type=float, default=0.2)  # 反向预测模型损失函数的比例为 0.2
    ap.add_argument("--Qloss_scale", type=float, default=0.1)  # Q 值损失函数比例为1
    ap.add_argument("--intrinsic_scale", type=float, default=1)  # 内在奖励的比例为 1
    ap.add_argument("--use_extrinsic", type=bool, default=True)

    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--veh_num", type=int, default=2983)

    ap.add_argument("--l2v", type=bool, default=False)

    ap.add_argument("--algo", type=str, default="nav", choices=["self_org", "hatt_router", "iql_b", "astar_dqn", "adaptive", "nav", "dso"])

    ap.add_argument("--threshold", type=int, default=10000)

    ap.add_argument("--road_feature", type=int, default=12)  # 6
    ap.add_argument("--cav_feature", type=int, default=11)  # 1
    ap.add_argument("--org", type=str, default="fixed")
    ap.add_argument("--router", type=bool, default=True)
    ap.add_argument("--use_attention", type=bool, default=True)
    ap.add_argument("--lighter", type=bool, default=False)
    ap.add_argument("--decay", type=bool, default=False)
    ap.add_argument("--map", type=str, default="hangzhou")
    ap.add_argument("--agent_num", type=int, default=16)

    ap.add_argument("--cop", type=str, default="independent", choices=["independent", "local", "global"])
    ap.add_argument("--control_type", type=str, default="syn", choices=["adapt", "fixed", "syn"])
    ap.add_argument("--phase_dim", type=int, default=2)
    ap.add_argument("--action_dim", type=int, default=2)
    ap.add_argument("--direction", type=int, default=3)
    # ap.add_argument("--action_embedding", type=int, default=2)
    ap.add_argument("--state_dim", type=int, default=38)  # state_dim = 16 + phase + phase_duration
    ap.add_argument("--global_state_dim", type=int, default=50 * 9)
    ap.add_argument("--counter_factual",type=float,default=1e-5)

    ap.add_argument("--batch_size", type=int, default=30)
    if EXEC:
        ap.add_argument("--episode", type=int, default=1)
        ap.add_argument("--buffer_size", type=int, default=0)
    else:
        ap.add_argument("--episode", type=int, default=2)
        ap.add_argument("--buffer_size", type=int, default=1000)
    ap.add_argument("--train_eps", type=int, default=35)
    ap.add_argument("--train_thr", type=int, default=0)
    ap.add_argument("--reward", type=str, default="length", choices=["length", "pressure", "waiting", "first_waiting"])
    ap.add_argument("--agent_type", type=str, default='PPO',
                    choices=['DQN', 'QMIX', 'DDPG', 'TD3', 'PPO', "MAT", "MAPPO", "MA2C", "CCGN", "CenPPO", "PS-PPO"])
    ap.add_argument("--spatial", type=str, default='FC', choices=['GAT', 'SAGE', "FC", "ORD_GAT"])
    ap.add_argument("--temporal", type=str, default='FC', choices=['GRU', 'LSTM', "FC"])
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--replay", type=str, default='UNI', choices=["UNI", "PER"])
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--state_contain_action", type=bool, default=False)
    ap.add_argument("--state_contain_phase", type=bool, default=True)
    ap.add_argument("--state_contain_phase_duration", type=bool, default=True)

    ap.add_argument("--rnn_hidden_dim", type=int, default=64)
    ap.add_argument("--obs_dim", type=int, default=50)
    ap.add_argument("--episode_limit", type=int, default=300)
    ap.add_argument("--hyper_hidden_dim", type=int, default=64)
    ap.add_argument("--qmix_hidden_dim", type=int, default=32)
    ap.add_argument("--lr", type=int, default=5e-4)
    ap.add_argument("--two_hyper_layers", type=bool, default=False)
    ap.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    ap.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')

    ap.add_argument('--tau', type=float, default=1e-3)

    return ap


def get_ppo_arguments(EXEC=False):
    ap = get_common_args(EXEC)
    ap.add_argument("--state_contain_agent_id", type=bool, default=True)
    ap.add_argument("--max_train_steps", type=int, default=int(3.8e4), help=" Maximum number of training steps")
    ap.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    ap.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    ap.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    ap.add_argument("--hidden_width", type=int, default=64,
                    help="The number of neurons in hidden layers of the neural network")
    ap.add_argument("--critic_hidden_width", type=int, default=100)
    ap.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")  # CenPPO 3e-4 PS-PPO 5e-4
    ap.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    ap.add_argument("--gamma", type=float, default=0.9, help="Discount factor")  # CenPPO 0.9 PS-PPO 0.99
    ap.add_argument("--lamda", type=float, default=1, help="GAE parameter")  # CenPPO 1 PS-PPO 0.97
    ap.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")  # CenPPO 0.2 PS-PPO 0.3
    ap.add_argument("--K_epochs", type=int, default=5, help="PPO parameter")
    ap.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
    ap.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    ap.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    ap.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    ap.add_argument("--entropy_coef", type=float, default=0.00, help="Trick 5: policy entropy")
    ap.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    ap.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    ap.add_argument("--max_grad_norm", type=float, default=10.0,
                    help='max norm of gradients (default: 0.5)')
    ap.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
    ap.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
    ap.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")

    ap.add_argument("--use_valuenorm", action='store_false', default=False,
                    help="by default True, use running mean and std to normalize rewards.")

    return ap


def get_mat_arguments(EXEC=False):
    ap = get_common_args(EXEC)
    ap.add_argument("--state_contain_agent_id", type=bool, default=True)
    ap.add_argument("--algorithm_name", type=str,
                    default='mat', choices=["mat", "mat_dec", "mat_encoder", "mat_decoder", "mat_gru"])

    ap.add_argument("--experiment_name", type=str, default="check",
                    help="an identifier to distinguish different experiment.")
    ap.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    ap.add_argument("--cuda", action='store_false', default=True,
                    help="by default True, will use GPU to train; or else will use CPU;")
    ap.add_argument("--cuda_deterministic",
                    action='store_false', default=True,
                    help="by default, make sure random seed effective. if set, bypass such function.")
    ap.add_argument("--n_training_threads", type=int,
                    default=1, help="Number of torch threads for training")
    ap.add_argument("--n_rollout_threads", type=int, default=1,
                    help="Number of parallel envs for training rollouts")
    ap.add_argument("--n_eval_rollout_threads", type=int, default=1,
                    help="Number of parallel envs for evaluating rollouts")
    ap.add_argument("--n_render_rollout_threads", type=int, default=1,
                    help="Number of parallel envs for rendering rollouts")
    ap.add_argument("--num_env_steps", type=int, default=10e6,
                    help='Number of environment steps to train (default: 10e6)')
    ap.add_argument("--user_name", type=str, default='xxx',
                    help="[for wandb usage], to specify user's name for simply collecting training data.")
    ap.add_argument("--use_wandb", action='store_false', default=False,
                    help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    ap.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    ap.add_argument("--use_obs_instead_of_state", action='store_true',
                    default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    ap.add_argument("--episode_length", type=int,
                    default=240, help="Max length for any episode")

    # network parameters
    ap.add_argument("--share_policy", action='store_false',
                    default=True, help='Whether agent share the same policy')
    ap.add_argument("--use_centralized_V", action='store_false',
                    default=True, help="Whether to use centralized V function")
    ap.add_argument("--stacked_frames", type=int, default=1,
                    help="Dimension of hidden layers for actor/critic networks")
    ap.add_argument("--use_stacked_frames", action='store_true',
                    default=False, help="Whether to use stacked_frames")
    ap.add_argument("--hidden_size", type=int, default=64,
                    help="Dimension of hidden layers for actor/critic networks")
    ap.add_argument("--layer_N", type=int, default=2,
                    help="Number of layers for actor/critic networks")
    ap.add_argument("--use_ReLU", action='store_false',
                    default=True, help="Whether to use ReLU")
    ap.add_argument("--use_popart", action='store_true', default=False,
                    help="by default False, use PopArt to normalize rewards.")
    ap.add_argument("--use_valuenorm", action='store_false', default=True,
                    help="by default True, use running mean and std to normalize rewards.")
    ap.add_argument("--use_feature_normalization", action='store_false',
                    default=True, help="Whether to apply layernorm to the inputs")
    ap.add_argument("--use_orthogonal", action='store_false', default=True,
                    help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    ap.add_argument("--gain", type=float, default=0.01,
                    help="The gain # of last action layer")

    # recurrent parameters
    ap.add_argument("--use_naive_recurrent_policy", action='store_true',
                    default=False, help='Whether to use a naive recurrent policy')
    ap.add_argument("--use_recurrent_policy", action='store_true',
                    default=False, help='use a recurrent policy')
    ap.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    ap.add_argument("--data_chunk_length", type=int, default=10,
                    help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    # ap.add_argument("--lr", type=float, default=5e-4,
    #                     help='learning rate (default: 5e-4)')
    ap.add_argument("--critic_lr", type=float, default=5e-4,
                    help='critic learning rate (default: 5e-4)')
    ap.add_argument("--opti_eps", type=float, default=1e-5,
                    help='RMSprop optimizer epsilon (default: 1e-5)')
    ap.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    ap.add_argument("--ppo_epoch", type=int, default=15,
                    help='number of ppo epochs (default: 15)')
    ap.add_argument("--use_clipped_value_loss",
                    action='store_false', default=True,
                    help="by default, clip loss value. If set, do not clip loss value.")
    ap.add_argument("--clip_param", type=float, default=0.2,
                    help='ppo clip parameter (default: 0.2)')
    ap.add_argument("--num_mini_batch", type=int, default=1,
                    help='number of batches for ppo (default: 1)')
    ap.add_argument("--entropy_coef", type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
    ap.add_argument("--value_loss_coef", type=float,
                    default=1, help='value loss coefficient (default: 0.5)')
    ap.add_argument("--use_max_grad_norm",
                    action='store_false', default=True,
                    help="by default, use max norm of gradients. If set, do not use.")
    ap.add_argument("--max_grad_norm", type=float, default=10.0,
                    help='max norm of gradients (default: 0.5)')
    ap.add_argument("--use_gae", action='store_false',
                    default=True, help='use generalized advantage estimation')
    ap.add_argument("--gamma", type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
    ap.add_argument("--gae_lambda", type=float, default=0.95,
                    help='gae lambda parameter (default: 0.95)')
    ap.add_argument("--use_proper_time_limits", action='store_true',
                    default=False, help='compute returns taking into account time limits')
    ap.add_argument("--use_huber_loss", action='store_false', default=True,
                    help="by default, use huber loss. If set, do not use huber loss.")
    ap.add_argument("--use_value_active_masks",
                    action='store_false', default=True,
                    help="by default True, whether to mask useless data in value loss.")
    ap.add_argument("--use_policy_active_masks",
                    action='store_false', default=True,
                    help="by default True, whether to mask useless data in policy loss.")
    ap.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    ap.add_argument("--use_linear_lr_decay", action='store_true',
                    default=False, help='use a linear schedule on the learning rate')
    # save parameters
    ap.add_argument("--save_interval", type=int, default=100,
                    help="time duration between contiunous twice models saving.")

    # log parameters
    ap.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    ap.add_argument("--use_eval", action='store_true', default=False,
                    help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    ap.add_argument("--eval_interval", type=int, default=25,
                    help="time duration between contiunous twice evaluation progress.")
    ap.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    ap.add_argument("--save_gifs", action='store_true', default=False,
                    help="by default, do not save render video. If set, save video.")
    ap.add_argument("--use_render", action='store_true', default=False,
                    help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    ap.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    ap.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    ap.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")

    # add for transformer
    ap.add_argument("--encode_state", action='store_true', default=False)
    ap.add_argument("--n_block", type=int, default=1)
    ap.add_argument("--n_embd", type=int, default=64)
    ap.add_argument("--n_head", type=int, default=1)
    ap.add_argument("--dec_actor", action='store_true', default=False)
    ap.add_argument("--share_actor", action='store_true', default=False)

    # add for online multi-task
    ap.add_argument("--train_maps", type=str, nargs='+', default=None)
    ap.add_argument("--eval_maps", type=str, nargs='+', default=None)

    ap.add_argument('--scenario', type=str, default='academy_3_vs_1_with_keeper')
    ap.add_argument('--n_agent', type=int, default=3)
    ap.add_argument("--add_move_state", action='store_true', default=False)
    ap.add_argument("--add_local_obs", action='store_true', default=False)
    ap.add_argument("--add_distance_state", action='store_true', default=False)
    ap.add_argument("--add_enemy_action_state", action='store_true', default=False)
    ap.add_argument("--add_agent_id", action='store_true', default=False)
    ap.add_argument("--add_visible_state", action='store_true', default=False)
    ap.add_argument("--add_xy_state", action='store_true', default=False)

    # agent-specific state should be designed carefully
    ap.add_argument("--use_state_agent", action='store_true', default=False)
    ap.add_argument("--use_mustalive", action='store_false', default=True)
    ap.add_argument("--add_center_xy", action='store_true', default=False)

    return ap


def get_mappo_arguments(EXEC=False):
    ap = get_common_args(EXEC)
    ap.add_argument("--state_contain_agent_id", type=bool, default=True)
    ap.add_argument("--algorithm_name", type=str,
                    default='mat', choices=["mat", "mat_dec", "mat_encoder", "mat_decoder", "mat_gru"])

    ap.add_argument("--hidden_sizes", type=list, default=[128, 128])
    ap.add_argument("--activation_func", type=str, default="relu")
    ap.add_argument("--initialization_method", type=str, default="orthogonal_")

    ap.add_argument("--experiment_name", type=str, default="check",
                    help="an identifier to distinguish different experiment.")
    ap.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    ap.add_argument("--seed_specify", type=bool, default=True)
    ap.add_argument("--cuda", action='store_false', default=True,
                    help="by default True, will use GPU to train; or else will use CPU;")
    ap.add_argument("--cuda_deterministic",
                    action='store_false', default=True,
                    help="by default, make sure random seed effective. if set, bypass such function.")
    ap.add_argument("--torch_threads", type=int, default=1)
    ap.add_argument("--n_training_threads", type=int,
                    default=1, help="Number of torch threads for training")

    ap.add_argument("--n_rollout_threads", type=int, default=1,
                    help="Number of parallel envs for training rollouts")
    ap.add_argument("--n_eval_rollout_threads", type=int, default=1,
                    help="Number of parallel envs for evaluating rollouts")
    ap.add_argument("--n_render_rollout_threads", type=int, default=1,
                    help="Number of parallel envs for rendering rollouts")
    ap.add_argument("--num_env_steps", type=int, default=10e6,
                    help='Number of environment steps to train (default: 10e6)')
    # ap.add_argument("--log_interval", type=int, default=5)
    # ap.add_argument("--eval_interval", type=int, default=25)
    ap.add_argument("--user_name", type=str, default='xxx',
                    help="[for wandb usage], to specify user's name for simply collecting training data.")
    ap.add_argument("--use_wandb", action='store_false', default=False,
                    help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    ap.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    ap.add_argument("--use_obs_instead_of_state", action='store_true',
                    default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    ap.add_argument("--episode_length", type=int,
                    default=240, help="Max length for any episode")

    # network parameters
    ap.add_argument("--share_policy", action='store_false',
                    default=True, help='Whether agent share the same policy')
    ap.add_argument("--use_centralized_V", action='store_false',
                    default=True, help="Whether to use centralized V function")
    ap.add_argument("--stacked_frames", type=int, default=1,
                    help="Dimension of hidden layers for actor/critic networks")
    ap.add_argument("--use_stacked_frames", action='store_true',
                    default=False, help="Whether to use stacked_frames")
    ap.add_argument("--hidden_size", type=int, default=64,
                    help="Dimension of hidden layers for actor/critic networks")
    ap.add_argument("--layer_N", type=int, default=2,
                    help="Number of layers for actor/critic networks")
    ap.add_argument("--use_ReLU", action='store_false',
                    default=True, help="Whether to use ReLU")
    ap.add_argument("--use_popart", action='store_true', default=False,
                    help="by default False, use PopArt to normalize rewards.")
    ap.add_argument("--use_valuenorm", action='store_false', default=True,
                    help="by default True, use running mean and std to normalize rewards.")
    ap.add_argument("--use_feature_normalization", action='store_false',
                    default=True, help="Whether to apply layernorm to the inputs")
    ap.add_argument("--use_orthogonal", action='store_false', default=True,
                    help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    ap.add_argument("--gain", type=float, default=0.01,
                    help="The gain # of last action layer")

    # recurrent parameters
    ap.add_argument("--use_naive_recurrent_policy", action='store_true',
                    default=False, help='Whether to use a naive recurrent policy')
    ap.add_argument("--use_recurrent_policy", action='store_true',
                    default=False, help='use a recurrent policy')
    ap.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    ap.add_argument("--recurrent_n", type=int, default=1, help="The number of recurrent layers.")
    ap.add_argument("--data_chunk_length", type=int, default=10,
                    help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    # ap.add_argument("--lr", type=float, default=5e-4,
    #                     help='learning rate (default: 5e-4)')
    ap.add_argument("--critic_lr", type=float, default=1e-3,
                    help='critic learning rate (default: 5e-4)')
    ap.add_argument("--opti_eps", type=float, default=1e-5,
                    help='RMSprop optimizer epsilon (default: 1e-5)')
    ap.add_argument("--weight_decay", type=float, default=0)

    ap.add_argument("--std_x_coef", type=float, default=1)
    ap.add_argument("--std_y_coef", type=float, default=0.5)

    # ppo parameters
    ap.add_argument("--ppo_epoch", type=int, default=5,
                    help='number of ppo epochs (default: 15)')
    ap.add_argument("--critic_epoch", type=int, default=5)
    ap.add_argument("--use_clipped_value_loss",
                    action='store_false', default=True,
                    help="by default, clip loss value. If set, do not clip loss value.")
    ap.add_argument("--clip_param", type=float, default=0.2,
                    help='ppo clip parameter (default: 0.2)')
    ap.add_argument("--num_mini_batch", type=int, default=1,
                    help='number of batches for ppo (default: 1)')
    ap.add_argument("--actor_num_mini_batch", type=int, default=1)
    ap.add_argument("--critic_num_mini_batch", type=int, default=1)
    ap.add_argument("--entropy_coef", type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
    ap.add_argument("--value_loss_coef", type=float,
                    default=1, help='value loss coefficient (default: 0.5)')
    ap.add_argument("--use_max_grad_norm",
                    action='store_false', default=True,
                    help="by default, use max norm of gradients. If set, do not use.")
    ap.add_argument("--max_grad_norm", type=float, default=10.0,
                    help='max norm of gradients (default: 0.5)')
    ap.add_argument("--use_gae", action='store_false',
                    default=True, help='use generalized advantage estimation')
    ap.add_argument("--gamma", type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
    ap.add_argument("--gae_lambda", type=float, default=0.95,
                    help='gae lambda parameter (default: 0.95)')
    ap.add_argument("--use_proper_time_limits", action='store_true',
                    default=True, help='compute returns taking into account time limits')
    ap.add_argument("--use_huber_loss", action='store_false', default=True,
                    help="by default, use huber loss. If set, do not use huber loss.")
    ap.add_argument("--use_value_active_masks",
                    action='store_false', default=True,
                    help="by default True, whether to mask useless data in value loss.")
    ap.add_argument("--use_policy_active_masks",
                    action='store_false', default=True,
                    help="by default True, whether to mask useless data in policy loss.")
    ap.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    ap.add_argument("--action_aggregation", type=str, default="prod")
    ap.add_argument("--share_param", type=bool, default=True)
    ap.add_argument("--fixed_order", type=bool, default=True)

    # run parameters
    ap.add_argument("--use_linear_lr_decay", action='store_true',
                    default=False, help='use a linear schedule on the learning rate')
    # save parameters
    ap.add_argument("--save_interval", type=int, default=100,
                    help="time duration between contiunous twice models saving.")

    # log parameters
    ap.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    ap.add_argument("--use_eval", action='store_true', default=False,
                    help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    ap.add_argument("--eval_interval", type=int, default=25,
                    help="time duration between contiunous twice evaluation progress.")
    ap.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    ap.add_argument("--save_gifs", action='store_true', default=False,
                    help="by default, do not save render video. If set, save video.")
    ap.add_argument("--use_render", action='store_true', default=False,
                    help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    ap.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    ap.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    ap.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")

    # add for transformer
    ap.add_argument("--encode_state", action='store_true', default=False)
    ap.add_argument("--n_block", type=int, default=1)
    ap.add_argument("--n_embd", type=int, default=64)
    ap.add_argument("--n_head", type=int, default=1)
    ap.add_argument("--dec_actor", action='store_true', default=False)
    ap.add_argument("--share_actor", action='store_true', default=False)

    # add for online multi-task
    ap.add_argument("--train_maps", type=str, nargs='+', default=None)
    ap.add_argument("--eval_maps", type=str, nargs='+', default=None)

    ap.add_argument('--scenario', type=str, default='academy_3_vs_1_with_keeper')
    ap.add_argument('--n_agent', type=int, default=3)
    ap.add_argument("--add_move_state", action='store_true', default=False)
    ap.add_argument("--add_local_obs", action='store_true', default=False)
    ap.add_argument("--add_distance_state", action='store_true', default=False)
    ap.add_argument("--add_enemy_action_state", action='store_true', default=False)
    ap.add_argument("--add_agent_id", action='store_true', default=False)
    ap.add_argument("--add_visible_state", action='store_true', default=False)
    ap.add_argument("--add_xy_state", action='store_true', default=False)

    # agent-specific state should be designed carefully
    ap.add_argument("--use_state_agent", action='store_true', default=False)
    ap.add_argument("--use_mustalive", action='store_false', default=True)
    ap.add_argument("--add_center_xy", action='store_true', default=False)

    return ap


def get_ma2c_arguments(EXEC=False):
    ap = get_common_args(EXEC)
    ap.add_argument("--state_contain_agent_id", type=bool, default=True)
    ap.add_argument("--rmsp_alpha", type=float, default=0.99)
    ap.add_argument("--rmsp_epsilon", type=float, default=1e-5)
    ap.add_argument("--max_grad_norm", type=float, default=40)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr_init", type=float, default=5e-4)
    ap.add_argument("--lr_decay", type=str, default="constant")
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--num_lstm", type=int, default=64)
    ap.add_argument("--num_fc", type=int, default=64)
    args = ap.parse_args()
    args.batch_size = 120
    args.batch_size = 120
    # ap.add_argument("--batch_size", type=float, default=120)

    ap.add_argument("--reward_norm", type=int, default=-1)
    ap.add_argument("--reward_clip", type=float, default=-1)

    ap.add_argument("--total_step", type=int, default=1e6)
    ap.add_argument("--test_interval", type=int, default=2e6)
    ap.add_argument("--log_interval", type=int, default=1e4)

    ap.add_argument("--clip_wave", type=float, default=2.0)
    ap.add_argument("--clip_wait", type=float, default=-1)
    ap.add_argument("--control_interval_sec", type=int, default=15)

    ap.add_argument("--agent", type=str, default="ma2c_nc")

    ap.add_argument("--coop_gamma", type=float, default=-1)
    ap.add_argument("--data_path", type=str, default="./")
    ap.add_argument("--episode_length_sec", type=float, default=3600)

    ap.add_argument("--norm_wave", type=float, default=5.0)
    ap.add_argument("--norm_wait", type=float, default=-1)
    ap.add_argument("--coef_wait", type=int, default=0)
    ap.add_argument("--peak_flow1", type=int, default=1100)
    ap.add_argument("--peak_flow2", type=int, default=925)
    ap.add_argument("--init_density", type=int, default=0)

    ap.add_argument("--objective", type=list, default=["queue", "wait"])
    ap.add_argument("--scenario", type=str, default="atsc_large_grid")
    # ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--test_seeds", type=int, default=10000)
    ap.add_argument("--yellow_interval_sec", type=int, default=3)

    return ap
