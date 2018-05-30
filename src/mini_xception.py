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

class Mini_Xception():
    def __init__(self, 
                 input_shape, 
                 emotion_map, 
                 learning_rate,
                 l2_regularization=0.0001):
        self.input_shape = input_shape
        self.l2_regularization = l2_regularization
        self._init_model()



    def _init_model(self):

        regularization = l2(self.l2_regularization)
        input_image = Input(shape=self.input_shape)

        # block1
        x = Conv2D(8, (3, 3), padding='same', use_bias=False, 
            kernel_regularizer=regularization, name='block1_conv1')(input_image)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv2D(8, (3, 3), padding='same', use_bias=False,
            kernel_regularizer=regularization, name='block1_conv2')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        # block2
        residual = Conv2D(16, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False,
                            name='block2_sepconv1')(x)
        x = BatchNormalization(name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False,
                            name='block2_sepconv2')(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
        x = add([x, residual])

        # block3
        residual = Conv2D(32, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False,
                            name='block3_sepconv1')(x)
        x = BatchNormalization(name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False,
                            name='block3_sepconv2')(x)
        x = BatchNormalization(name='block3_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(
            2, 2), padding='same', name='block3_pool')(x)
        x = add([x, residual])
    
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
        x = add([x, residual])

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
        x = add([x, residual])

        x = Conv2D(512, (4, 4), strides=(1, 1), padding='same', 
                   name='block6_conv')(x)
        x = Flatten(name='flatten')(x)
        prediction_layer = Dense(len(self.emotion_map), activation='softmax',
                                 name='prediction')(x)
        model = Model(input=input_image, output=prediction_layer)

        self.model = model



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