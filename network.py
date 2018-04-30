from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, GlobalAveragePooling2D,
    Conv2D, MaxPooling3D, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils import plot_model
import json

def Transfer_learing_VGG16(input_shape, num_classes):
    conv_based_model = VGG16(include_top=False, weights='imagenet', 
        input_shape=input_shape)
    top_lay_model = conv_based_model.output
    top_lay_model = GlobalAveragePooling2D()(top_lay_model)
    top_lay_model = Dropout(0.5)(top_lay_model)
    top_lay_model = Dense(1024, activation='relu')(top_lay_model)
    top_lay_model = Dropout(0.5)(top_lay_model)
    top_lay_model = Dense()