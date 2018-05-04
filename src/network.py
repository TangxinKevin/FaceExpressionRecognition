from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger,\
                            ModelCheckpoint
from keras.layers import Dense, Flatten, GlobalAveragePooling2D,
    Conv2D, MaxPooling3D, Input, Dropout
from keras.layers import SeparableConv2D, Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model, Sequential
from keras import optimizers
from keras.utils import plot_model
from src.callback import PlotLosses
import json
import os

class _FERNetwork():
    def __init__(self, emotion_map):
        self.emotion_map = emotion_map
        self._init_model()
    
    def _init_model(self):
        raise NotImplementedError("Class %s doesn't implement _init_ \
            model()" % self.__class__.__name__)
    
    def fit(self, x_train, y_train):
        raise NotImplementedError(
            "Class %s doesn't implement fit()" % self.__class__.__name__)

    def fit_generator(self, generator, learning_rate, dataset_name,
                      log_file_path, model_path, 
                      validation_data=None, epochs=50):
        log_file = os.path.join(log_file_path, self.__class__ + '_' + dataset_name
                                + '_emotion_training.log')
        csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5,
                                      patience=5, verbose=1)
        model_name = os.path.join(model_path, dataset_name + '_' 
                                  + self.__class__ + '.{epoch:02d}-{val_acc:.2f}.hdf5')
        model_checkpoint = ModelCheckpoint(model_name, 'val_loss', verbose=1,
                                           save_best_only=True)
        callbacks = [model_checkpoint, csv_logger, early_stop, PlotLosses,
                     reduce_lr]
        rmsprop = optimizers.RMSprop(lr=learning_rate)
        self.model.compile(optimize=rmsprop, loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit_generator(generator=generator,
                                 validation_data=validation_data, 
                                 epochs=epochs,
                                 callbacks=callbacks)

    def predict(self, images):
        self.model.predict(images)
    
    def save_model_path(self):
        plot_model(self.model, to_file='output/model.png')
    
    def export_model(self, model_filepath, weights_filepath, 
                     emotion_map_filepath, emotion_map):
        self.model.save_weights(weights_filepath)

        model_json_string = self.model.to_json()

        with open(model_filepath, 'w') as f:
            f.write(model_json_string)
        
        with open(emotion_map_filepath, 'w') as f_json:
            json.dump(emotion_map, fp)
    
    def import_model_weights(self, weights_filepath):
        self.model.load_weights(weights_filepath, by_name=True)


class TransferLearningDeepCNN(_FERNetwork):
    """
    Transfere Learning Deep Convolutional Neural Network initialized
    with pretrained weights.
    
    # param:
        model_name: name of pretrained model to use for initial weights.
            options: ['Xception', 'VGG16', 'VGG19', 'ResNet50', 
                      'InceptionV3',' InceptionResnetV2']
        emotion_map: dict of target emotion label keys with int values 
           corresponding to the index of the emotion probability in the
           prediction output array
    
    # Example:
        model = TransferLearningDeepCNN(name='InceptionV3', 
                                        target_labels=[0,1,2,3,4,5,6])
        model.fit(images, labels, validation_split=0.15)
    """
    _NUM_BOTTOM_LAYERS_TO_RETURN = 249

    def __init__(self, num_layers, 
                 input_shape, 
                 model_name, 
                 emotion_map):
        self.num_layers_untrainable = num_layers
        self.model_name = model_name
        self.input_shape = input_shape
        super(TransferLearningDeepCNN, self).__init__(emotion_map)
    
    def _init_model(self):
        """
        Initialize base model from keras and add top layers to match 
            number of training emotions labels.
        """
        base_model = self._get_base_model()

        top_layer_model = base_model.output
        top_layer_model = GlobalAveragePooling2D\
            (name='global_average_pooling')(top_layer_model)
        top_layer_model = Dropout(0.5, name='dropout1')(top_layer_model)
        top_layer_model = Dense(1024, activation='relu', name='fc1')\
            (top_layer_model)
        top_layer_model = Dropout(0.5, name='dropout2')(top_layer_model)
        prediction_layer = Dense(len(self.emotion_map.keys()), 
            activation='softmax', name='prediction')(top_layer_model)
        
        model = Model(input=base_model.input,
                      output=prediction_layer)
        print(model.summary())
        
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimize='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.mdoel = model

    
    def _get_base_model(self):
        """
        return: base model on user-supplied model name
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
        elif self.model_name == 'Xception':
            return Xception(include_top=False,
                            weights='imagenet',
                            input_shape=self.input_shape)
        else:
            ValueError("Cannot find base model %s" % self.model_name)
    
    def fit(self, features, labels, validation_split, epochs=50):
        """
        Trains the neural net on the data provided.

        # param:
            features: Numpy array of training data.
            labels: Numpy array of target data.
            validation_split: Float between 0 and 1. Percentage of training
                data to use for validation
            epochs: Max number of times to train over dataset.
        """     
        for layer in self.model.layers[:self.num_layers_untrainable]:
            layer.trainbale = False
        for layer in self.model.layers[self.num_layers_untrainable:]:
            layer.trainable = True
        
        self.model.compile(optimize='rmsprop', loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(x=features, y=labels, epochs=50, verbose=1,
                       callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=5)],
                       validation_split=validation_split, shuffle=True)
    
class ConvolutionalNN(_FERNetwork):
    """
    2D Convolutional Neural Network

    # param:
        input_shape: the shape of image
        emotion_map: dict of target emotion label keys
        filters_list: list of filters per layer in CNN
        kernel_size: size of sliding window for each layer of CNN
    """
    def __init__(self, input_shape, emotion_map, kernel_size=(3, 3)):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        super(ConvolutionalNN, self).__init__(emotion_map)
    
    def _init_model(self):
        """
        Composes all layers of 2D CNN.
        """
        input_image = Input(shape=self.input_shape)
        x = Conv2D(64, kernel_size=self.kernel_size, activation='relu')(input_image)
        x = Conv2D(64, kernel_size=self.kernel_size, activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Conv2D(128, kernel_size=self.kernel_size, activation='relu')(x)
        x = Conv2D(128, kernel_size=self.kernel_size, activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Conv2D(64, kernel_size=self.kernel_size, activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.7)(x)
        prediction_layer = Dense(len(self.emotion_map.keys()), activation='softmax')(x)
        
        model = Model(input=input_image, output=prediction_layer)
        model.summary()
        self.model = model
        
    def fit(self, image_data, labels, validation_split, epochs=50):
        """
        Trains the neural net on the data provided.
        """
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                            metrics=['accuracy'])
        self.model.fit(image_data, labels, epochs=epochs, validation_split=validation_split,
                        callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=5)])

class Mini_Xception(_FERNetwork):
    def __init__(self, input_shape, l2_regularization=0.0001, emotion_map):
        self.input_shape = input_shape
        self.l2_regularization = l2_regularization
        super(Mini_Xception, self).__init__(emotion_map)

    def _init_model(self):

        regularization = l2(self.l2_regularization)
        input_image = Input(shape=self.input_shape)

        # block1
        x = Conv2D(16, (3, 3), padding='same', use_bias=False, 
            kernel_regularizer=regularization, name='block1_conv1')(input_image)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv2D(16, (3, 3), padding='same', use_bias=False,
            kernel_regularizer=regularization, name='block1_conv2')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        # block2
        residual = Conv2D(32, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False,
                            name='block2_sepconv1')(x)
        x = BatchNormalization(name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False,
                            name='block2_sepconv2')(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
        x = Add([x, residual])

        # block3
        residual = Conv2D(64, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False,
                            name='block3_sepconv1')(x)
        x = BatchNormalization(name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False,
                            name='block3_sepconv2')(x)
        x = BatchNormalization(name='block3_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(
            2, 2), padding='same', name='block3_pool')(x)
        x = Add([x, residual])
    
        # block4
        residual = Conv2D(64, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False,
                            name='block4_sepconv1')(x)
        x = BatchNormalization(name='block4_sepconv1_bn')(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False,
                            name='block4_sepconv2')(x)
        x = BatchNormalization(name='block4_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(
            2, 2), padding='same', name='block4_pool')(x)
        x = Add([x, residual])

        # block5
        residual = Conv2D(128, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                            name='block5_sepconv1')(x)
        x = BatchNormalization(name='block5_sepconv1_bn')(x)
        x = Activation('relu', name='block5_sepconv2_act')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                            name='block5_sepconv2')(x)
        x = BatchNormalization(name='block5_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block5_pool')(x)
        x = Add([x, residual])

        x = Conv2D(512, (4, 4), strides=(1, 1), padding='same', 
                   name='block6_conv')(x)
        x = Flatten(name='flatten')(x)
        prediction_layer = Dense(len(self.emotion_map), activation='softmax',
                                 name='prediction')(x)
        model = Model(input=input_image, output=prediction_layer)

        self.model = model

class Varaint_VGG(_FERNetwork):
    def __init__(self, input_shape, emotion_map):
        self.input_shape = input_shape
        super(Varaint_VGG, self).__init__(emotion_map)

    def _init_model(self):
        input_image = Input(shape=self.input_shape)

        x = Conv2D(64, (3, 3),activation='relu')(input_image)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x) 
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        prediction_layer = Dense(len(self.emotion_map), activation='softmax')(x)

        model = Model(input=input_image, output=prediction_layer)

        self.model = model
    









        

        




