import numpy as np
from keras.utils  import Sequence
import keras.backend as K

class NumpyArrayGenerator(Sequence):
    """
    Generator for numpy array
    """
    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None):
        self.image_data_generator = image_data_generator
        self.x = np.asarray(x, dtype=K.floatx())
        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayGenerator`'
                             'should have rank 4. You passed an array with'
                             'shape', self.x.shape)
        self.y = np.asarray(y)
        self.n = x.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_array = None
        self.index_generator = self._flow_index()
    
    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx},'
                             'but the Squence ',
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size -1) // self.batch_size

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self): 
        self.batch_index = 0
    
    def _flow_index(self):
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]
    
    def _get_batches_of_transformed_samples(self, index_array):
        """
        Gets a batch of transformed samples.
        """
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape[1:])), 
                           dtype=K.floatx())
        batch_y = np.zeros(tuple([len(index_array)] + list(self.y.shape[1])),
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(
                x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = self.y[j]
        return batch_x, batch_y


class DirectoryGenerator(Sequence):
    """
    Generator for numpy array
    """

    def __init__(self, data_file, label, image_data_generator,
                 target_image_size, out_channels,
                 batch_size=32, shuffle=False, seed=None):
        self.image_data_generator = image_data_generator
        self.data_file = data_file
        self.label = np.asarray(label, dtype=K.floatx())
        self.target_image_size = target_image_size
        self.out_channels = out_channels

        self.n = len(data_file)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx},'
                             'but the Squence ',
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def _get_batches_of_transformed_samples(self, index_array):
        """
        Gets a batch of transformed samples.
        """
        batch_x = np.zeros(tuple([len(index_array)] + list(self.target_image_size) 
                           + [self.out_channels]), dtype=K.floatx())
        batch_y = np.zeros(tuple([len(index_array)] + list(self.label.shape[1])),
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.load_image(self.data_file[j])
            x = self.image_data_generator.random_transform(
                x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = self.label[j]
        return batch_x, batch_y

    def load_image(self, image_path):
        if self.out_channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imrerad(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.BGR2RGB)

        if image.shape[0] != self.target_image_size[0] or
            image.shape[1] != self.target_image_size[1]:
            image = cv2.resize(image, self.target_image_size,
                               interpolation=cv2.INTER_CUBIC)
        return image
