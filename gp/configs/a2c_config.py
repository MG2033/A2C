from gp.a2c.models.cnn_policy import CNNPolicy
from gp.a2c.envs.gym_env import GymEnv
from gp.utils.utils import create_experiment_dirs


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
    load = True

    # Summaries Config
    scalar_summary_tags = []
    scalar_summary_tags.extend(['policy-loss', 'policy-entropy', 'value-function-loss', 'reward'])

    summary_dir, checkpoint_dir = create_experiment_dirs('../../a2c/experiments/' + experiment_dir)
