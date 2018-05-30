from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Flatten, GlobalAveragePooling2D 
from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Activation
from keras.layers import SeparableConv2D, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model, Sequential
from keras import optimizers
import json
import os

class DeepTransferLearningNet(object):
    """
    Pre-trained model 
    """
    def __init__(self, 
                 input_shape,
                 model_name,
                 emotion_map,
                 learning_rate):
        self.input_shape = input_shape
        self.model_name = model_name
        self.emotion_map = emotion_map
        self.leaning_rate = learning_rate

        self._init_model()

    def _init_model(self):
        base_model = self._get_base_model()
        top_layer_model = base_model.output
        top_layer_model = GlobalAveragePooling2D(
            name='global_average_pooling')(top_layer_model)
        top_layer_model = Dropout(0.5, name='dropout1')(top_layer_model)
        top_layer_model = Dense(1024, Activation='relu', name='fc1')(
            top_layer_model)
        top_layer_model = Dropout(0.5, name='dropout2')(top_layer_model)
        top_layer_model = Dense(len(self.emotion_map.keys()), 
            activation='softmax', name='pred')(top_layer_model)

        model = Model(input=base_model.input,
                      output=top_layer_model)
        model.summary()

        for layer in base_model.layers:
            layer.trainable = False

        self.model = model


    def _get_base_model(self):
        """
        Get the base model from pre-trainded model
        """
        if self.model_name == 'InceptionV3':
            return InceptionV3(include_top=False,
                               weights='imagenet',
                               input_shape=self.input_shape)
        elif self.model_name == 'InceptionResnetV2':
            return InceptionResNetV2(include_top=False,
                                     weights='imagenet',
                                     input_shape=self.input_shape)
        elif self.model_name == 'ResNet50':
            return ResNet50(include_top=False,
                            weights='imagenet',
                            input_shape=self.input_shape)
        elif self.model_name == 'VGG16':
            return VGG16(include_top=False,
                         weights='imagenet',
                         input_shape=self.input_shape)
        elif self.model_name == 'VGG19':
            return VGG19(include_top=False,
                         weights='imagenet',
                         input_shape=self.input_shape)
        elif self.model_name == 'Xception'：
            return Xception(include_top=False,
                            weights='imagenet',
                            input_shape=self.input_shape)
        else:
            ValueError("Cannot find base model %s".format(self.model_name))


    def fit(self, features, labels, validation_split, epochs=50):
        """
        Train the neural net on the data provided.

        # param:
            features: Numpy array of training data.
            labels: Numpy array of target data.
            validation_split: Float between 0 and 1. Percentage of training
                data to use for validation.
            epochs: Max number of times to train over dataset.
        """

        self.model.compile(optimizers='rmsprop', loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(Xception=features, y=labels, epochs=50, verbose=-1,
                       callbacks[ReduceLROnPlateau(), EarlyStopping(patience=20)],
                       validation_split=validation_split, shuffle=True)


    def fit_generator(self, generator, dataset_name, log_file_path, model_path,
                      validation_data=None, epochs=50):
        log_file = os.path.join(log_file_path, self.__class__.__name__ +
                                "_" + dataset_name + '_emotion_training.log')
        csv_logger = CSVLogger(log_file, append=False)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                      patience=20, min_lr=0.000001)
        model_name = os.path.join(model_path, dataset_name + '_' +
                                  self.__class__.__name__ + 
                                  '.{epoch:02d}-{val_acc:.2f}.hdf5')
        model_checkpoint = ModelCheckpoint(model_name, 'val_acc', verbose=1,
                                           save_best_only=True)
        callbacks = [model_checkpoint, csv_logger, reduce_lr]
        adam = optimizers.Adam(lr=self.learning_rate)

        self.model.compile(optimizers=adam,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit_generator(generator=generator,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=validation_data)


    def predict(self, images):
        return self.model.predict(images)


    def evaluate(self, images, labels):
        return self.model.evaluate(images, labels)


    def export_model(self, model_filepath, weights_filepath,
                     emotion_map_filepath, emotion_map):
        self.model.save_weights(weights_filepath)
        model_json_string = self.model.to_json()
        
        with open(model_file_path, 'w') as f:
            f.write(model_json_string)

        with open(emotion_map_filepath, 'w') as f_json:
            json.dump(emotion_map, f_json)


    def import_model_weights(self, weights_filepath):
        self.model.load_weights(weights_filepath, by_name=True)
    
