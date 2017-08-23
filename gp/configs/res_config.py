

class ResConfig:
    truncated_time_steps = 15
    state_size = 19
    dropout_rate = 0.3
    data_size = None
    batch_size = 16
    action_dim = 3
    learning_rate = .0001
    max_to_keep = 5  # checkpoints
    summary_dir = "/shared/Windows1/oabdelta/with_rewards_summaries/"
    load = True
    nit_epoch = 400
    n_epochs = 1500
    num_episodes_train = 2000
    num_episodes_test = 400
    num_episodes = 2400
    test_every = 5
    all_seq_length = 45
    epsilon = 0.15
    scalar_summary_tags = ['loss', 'sensor1_relative_error', 'test_MSE',
                           'sensor2_relative_error', 'sensor3_relative_error', 'sensor4_relative_error',
                           'sensor5_relative_error', 'sensor6_relative_error', 'sensor7_relative_error',
                           'sensor8_relative_error', 'sensor9_relative_error', 'sensor10_relative_error',
                           'sensor11_relative_error', 'sensor12_relative_error', 'sensor13_relative_error',
                           'sensor14_relative_error', 'sensor15_relative_error', 'sensor16_relative_error',
                           'sensor17_relative_error', 'sensor18_relative_error', 'sensor19_relative_error',
                           'rewards_relative_error']
    predict_reward = True
    checkpoint_dir = '/shared/Windows1/oabdelta/with_rewards_summaries/'
    x_path = '/shared/Windows1/oabdelta/new-torcs-data/states.npy'
    actions_path = '/shared/Windows1/oabdelta/new-torcs-data/actions.npy'
    rewards_path = '/shared/Windows1/oabdelta/new-torcs-data/rewards.npy'
    dyna_iterations = 30
