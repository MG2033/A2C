class I2AConfig:
    # input config
    frame_w, frame_h, frame_c = 15, 19, 3
    actions_num = 4
    action_space=[0,1,2,3]
    # trainig config
    batch_size = None
    rollouts_steps = 5

    # model config
    cnn_kernal_sizes = [8, 4, 3]
    cnn_strides = [4, 2, 1]
    cnn_kernals_num = [32, 64, 64]
    lstm_units = 512