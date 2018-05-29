from src.data_generator import NumpyArrayGenerator,\
                               DirectoryGenerator
from src.dataset import NovaEmotions
from src.data_utils import split_dataset
from src.network import TransferLearningDeepCNN, \
                        ConvolutionalNN, Mini_Xception
from config import DefaultConfig

from keras.preprocessing.image import ImageDataGenerator

#default parameters
opt = DefaultConfig()
data_path = '/home/user/Documents/dataset/NovaEmotions'


# Load dataset
dataset = NovaEmotions(opt.target_emotion_map, data_path)
images, labels, emotion_map = dataset.load_data()


# split_dataset
validation_split = 0.1
trainset, validationset = split_dataset(images, labels, validation_split)

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=30,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.1,
                                           zoom_range=0.1,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
test_image_generator = ImageDataGenerator(rescale=1./255)

train_gen = DirectoryGenerator(trainset[0], trainset[1], train_image_generator,
                               target_image_size=opt.target_image_size,
                               out_channels=opt.out_channels,
                                batch_size=opt.batch_size, shuffle=True,
                                seed=None)
validation_gen = DirectoryGenerator(validationset[0], validationset[1], test_image_generator,
                                    target_image_size=opt.target_image_size,
                                    out_channels=opt.out_channels,
                                    batch_size=opt.batch_size, shuffle=True,
                                    seed=None)

model = Mini_Xception(opt.input_shape, opt.target_emotion_map, 
                      opt.learning_rate, opt.l2_regularization)

model.fit_generator(train_gen, 'NovaEmotions', 
                    opt.log_file_path, opt.model_path,
                    validation_data=validation_gen, 
                    epochs=opt.epochs)
