from gp.a2c.models.cnn_policy import CNNPolicy
from gp.a2c.envs.gym_env import GymEnv


class A2CConfig:
    num_envs = 5
    env_class = GymEnv
    env_name = "SpaceInvaders"
    env_seed = 42
    policy_class = CNNPolicy
    unroll_time_steps = 5
    num_stack = 4
    num_iterations = 4e6
    learning_rate = 7e-4
    reward_discount_factor = 0.99
    max_to_keep = 10
    experiment_dir = "exp1"
    is_train = True
    cont_train = True
