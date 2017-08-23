class I2AConfig:
    # input config
    frame_w, frame_h, frame_c = None, None, None
    actions_num = None

    # trainig config
    batch_size = None
    rollouts_steps = None

    # model config
    cnn_kernal_sizes = [8, 4, 3]
    cnn_strides = [4, 2, 1]
    cnn_kernals_num = [32, 64, 64]
    fc_units = 512