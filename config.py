class DefaultConfig():
    target_emotion_map = {'0': 'anger', '1': 'fear',
                          '2': 'happy', '3': 'sadness', '4': 'surprise',
                          '5': 'neutral'}
    target_image_size = (64, 64)
    out_channels = 3
    batch_size = 64
    input_shape = (64, 64, 3)
    l2_regularization = 0.001
    learning_rate = 0.001
    log_file_path = '../log/'
    model_path = '../model/'
    epochs = 1000
