class ResConfig:
    # model configs
    predict_reward = True
    truncated_time_steps = 15
    state_size = [96, 96, 1]
    dropout_rate = 0.3
    lstm_size = 1024
    data_size = None
    batch_size = 4
    action_dim = 3
    learning_rate = .00001
    max_to_keep = 5  # checkpoints
    load = True
    is_train = True
    predict_reward = True
    # training configs
    nit_epoch = 18
    n_epochs = 10000
    num_episodes_train = 40
    num_episodes_test = 10
    num_episodes = 50
    episode_length = 45
    test_every = 1
    epsilon = 0.15
    observation_steps_length = 8
    env_id = 'Pong'
    # summaries configs
    scalar_summary_tags = ['loss', 'test_MSE']
    # test
    test_steps = 10
    # paths configs
    checkpoint_dir = './experiments/checkpoints/checkpoints'
    states_path = './data/states.npy'
    actions_path = './data/actions.npy'
    rewards_path = './data/rewards.npy'
    summary_dir = "./experiments/summaries"

    save_every = 10
