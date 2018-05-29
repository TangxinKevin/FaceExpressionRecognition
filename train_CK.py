from src.data_generator import NumpyArrayGenerator
from src.dataset import CK
from src.data_utils import split_dataset
from src.network import TransferLearningDeepCNN, \
                        ConvolutionalNN, Mini_Xception
from config import DefaultConfig
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)

#default parameters
opt = DefaultConfig()
data_path = '/home/user/Documents/dataset/Ck+'


# Load dataset
dataset = CK(opt.target_emotion_map, opt.target_image_size,
             opt.out_channels, data_path, detect_face=True)
images, vetorized_labels, labels, emotion_map = dataset.load_data()
print(labels)
images /= 255.

train_image_generator = ImageDataGenerator(featurewise_center=False,
                                           samplewise_center=False,
                                           rotation_range=10,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.1,
                                           zoom_range=0.1,
                                           horizontal_flip=True,
                                           fill_mode='nearest')

#model = ConvolutionalNN(opt.input_shape, opt.target_emotion_map, 
#                      opt.learning_rate)
#model = Mini_Xception(opt.input_shape, opt.target_emotion_map, 
#                      opt.learning_rate, opt.l2_regularization)
model = TransferLearningDeepCNN(249, opt.input_shape, "VGG16",
    opt.target_emotion_map, opt.learning_rate)
# Cross-validation
validation_scores = []
for train, test in skf.split(images, labels):
    train_gen = NumpyArrayGenerator(images[train], vetorized_labels[train], 
                                    train_image_generator,
                                    batch_size=opt.batch_size, shuffle=True,
                                    seed=None)
    model.fit_generator(train_gen, 'CK', 
                        opt.log_file_path, opt.model_path,
                        validation_data=(images[test], vetorized_labels[test]), 
                        epochs=opt.epochs)
    validation_score = model.evaluate(images[test], vetorized_labels[test])
    print(validation_score)
    validation_scores.append(validation_score[1])

cross_validation_socre = np.average(validation_scores)
print("The 10-fold cross validation accuracy is ", cross_validation_socre)
