from gp.a2c.models.cnn_policy import CNNPolicy
from gp.a2c.envs.gym_env import GymEnv
from gp.utils.utils import create_experiment_dirs


class A2CConfig:
    num_envs = 1
    env_class = GymEnv
    env_name = "PongNoFrameskip-v4"
    env_seed = 42
    policy_class = CNNPolicy
    unroll_time_steps = 5
    num_stack = 4
    num_iterations = 4e6
    learning_rate = 7e-4
    reward_discount_factor = 0.99
    max_to_keep = 4
    experiment_dir = "experiment_pong"
    is_train = True
    load = True

    # env_name="BreakoutNoFrameskip-v4"
    # max_to_keep = 4
    # experiment_dir = "experiment_1"
    # num_iterations = 10e6



    # If this equals -1, it means don't record
    record_video_every = -1

    # Summaries Config
    env_summary_tags = ['reward', 'episode_length']

    # Prepare Directories
    experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir = create_experiment_dirs(
        '../../a2c/experiments/' + experiment_dir)
