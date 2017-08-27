class ResConfig:
    #model configs
    predict_reward = True
    truncated_time_steps = 3
    state_size = [256, 160, 3]
    dropout_rate = 0.3
    lstm_size = 1024
    data_size = None
    batch_size = 4
    action_dim = 3
    learning_rate = .00001
    max_to_keep = 5  # checkpoints
    load = True
    predict_reward = True

    #training configs
    nit_epoch = 5
    n_epochs = 1500
    num_episodes_train = 10
    num_episodes_test = 400
    num_episodes = 10
    episode_length=10
    test_every = 5
    epsilon = 0.15

    #summaries configs
    scalar_summary_tags = ['loss', 'test_MSE']

    #paths configs
    checkpoint_dir = '/home/gemy/work/projects/GP/with_rewards_summaries/'
    x_path = '/home/gemy/work/projects/GP/states.npy'
    actions_path = '/home/gemy/work/projects/GP/actions.npy'
    rewards_path = '/home/gemy/work/projects/GP/rewards.npy'
    summary_dir = "/home/gemy/work/projects/GP/with_rewards_summaries/"
