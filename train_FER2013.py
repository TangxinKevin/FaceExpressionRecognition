from keras.preprocessing.image import ImageDataGenerator
from src.data_generator import NumpyArrayGenerator
from src.data_generator import DirectoryGenerator
from src.dataset import FER2013
from src.data_utils import split_dataset

from src.deepTransferLearningNet import DeepTransferLearningNet 
from src.convNet import ConvNet
from src.variantVGG import VariantVGG
from src.mini_xception import Mini_Xception

from config import DefaultConfig


#default parameters
opt = DefaultConfig()
raw_image_size = (48, 48)
data_csv_path = '/home/user/Documents/dataset/fer2013/fer2013.csv'


# Load dataset
dataset = FER2013(opt.target_emotion_map, raw_image_size, opt.target_image_size,
                  opt.out_channels, data_csv_path)
images, labels, emotion_map = dataset.load_data()


# split_dataset
validation_split = 0.1
trainset, validationset = split_dataset(images, labels, validation_split)

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=10,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.1,
                                           zoom_range=0.1,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
test_image_generator = ImageDataGenerator(rescale=1./255)

train_gen = NumpyArrayGenerator(trainset[0], trainset[1], train_image_generator,
                                batch_size=opt.batch_size, shuffle=True,
                                seed=None)
validation_gen = NumpyArrayGenerator(validationset[0], validationset[1],
                                     test_image_generator,
                                     batch_size=opt.batch_size)

model = Mini_Xception(opt.input_shape, opt.target_emotion_map, opt.l2_regularization)

model.fit_generator(train_gen, 
                    opt.learning_rate, 
                    'fer2013', 
                    opt.log_file_path, 
                    opt.model_path,
                    validation_data=validation_gen, 
                    epochs=opt.epochs)
