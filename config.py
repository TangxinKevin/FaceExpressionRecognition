class DefaultConfig():
    target_emotion_map = {'anger': 0, 'fear': 1,
                          'happy': 2, 'sadness': 3,
                          'surprise': 4, 'neutral': 5}
    target_image_size = (64, 64)
    out_channels = 3
    batch_size = 64
    input_shape = (64, 64, 3)
    l2_regularization = 0.001
    learning_rate = 0.0001
    log_file_path = '/home/user/Documents/delta/expression/FaceExpressionRecognition-master/log'
    model_path = '/home/user/Documents/delta/expression/FaceExpressionRecognition-master/model'
    epochs = 1000
