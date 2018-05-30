from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Activation
from keras.models import Model
from keras import optimizers
import os 
import json

class ConvNet(object):
    """
    2D Convolutional Neural Network
    """
    def __init__(self, 
                 input_shape,
                 emotion_map,
                 learning_rate,
                 kernel_size=(3, 3)):
        self.input_shape = input_shape
        self.emotion_map = emotion_map
        self.leaning_rate = learning_rate
        self.kernel_size = kernel_size

        self._init_model()

    def _init_model(self):
        """
        Compose all layers of 2D CNN.
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
    

