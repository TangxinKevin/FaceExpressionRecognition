import numpy as np
import csv
import cv2
from src.data_utils import split_dataset
import os
from glob import glob
import dlib
import pandas as pd
import keras.backend as K


class FER2013():
    def __init__(self, 
                 target_emotion_map,
                 row_image_size,
                 target_image_size, 
                 out_channels, 
                 data_csv_path):
        self.target_emotion_map = target_emotion_map
        self.row_image_size = row_image_size
        self.target_image_size = target_image_size
        self.out_channels = out_channels
        self.data_csv_path = data_csv_path
        self.FER_EMOTION = {'0': 'anger', '1': 'disgust', '2': 'fear',
                            '3': 'happy', '4': 'sadness', '5': 'surprise',
                            '6': 'neutral'} 

    def load_data(self):
        """
        Loads image and label data from specified csv file path.
        """
        print('Extracting training data from csv...')
        images = list()
        labels = list()
        emotion_index_map = self.target_emotion_map
        data = pd.read_csv(self.data_csv_path)
        emotion = data['emotion'].tolist()
        pixels = data['pixels'].tolist()
        for emo, pixel in zip(emotion, pixels):
            label_class = self.FER_EMOTION[str(emo)]
            if label_class not in self.target_emotion_map.keys():
                continue
            labels.append(label_class)
            image = np.asarray([int(i) for i in pixel.split(' ')],
                               dtype=np.uint8).reshape(self.row_image_size)
            image = self._reshape(image)
            images.append(image)
        vectorized_labels = self._vectorize_labels(emotion_index_map, labels)
        return (np.array(images), np.array(vectorized_labels), emotion_index_map)

    def _reshape(self, image):
        if image.shape[0] != self.target_image_size[0] or \
            image.shape[1] != self.target_image_size[1]:
            image = cv2.resize(image, self.target_image_size, 
                               interpolation=cv2.INTER_CUBIC)              
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        channels = image.shape[-1]

        if channels == 3 and self.out_channels == 1:
            gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
            return np.expand_dims(gray, axis=2)
        if channels == 1 and self.out_channels == 3:
            return np.repeat(image, repeats=3, axis=2)
        return image
    
    def _vectorize_labels(self, label_index_map, labels):
        label_values = list()
        label_count = len(label_index_map.keys())
        for label in labels:
            label_value = [0] * label_count
            label_value[label_index_map[label]] = 1.0
            label_values.append(label_value)
        return label_values


class CK():
    def __init__(self, target_emotion_map, target_image_size, out_channels,
                 data_path, detect_face=False):
        self.target_emotion_map = target_emotion_map
        self.target_image_size = target_image_size
        self.out_channels = out_channels
        self.data_path = data_path
        self.detect_face = detect_face
        self.image_path = os.path.join(self.data_path, 'cohn-kanade-images')
        self.label_path = os.path.join(self.data_path, 'Emotion')
        if self.detect_face:
            self.detector = dlib.get_frontal_face_detector()
        self.ck_emotion_map = {'0': 'neutral', '1': 'anger', '2': 'contempt',
                               '3': 'disgust', '4': 'fear', '5': 'happy',
                               '6': 'sadness', '7': 'surprise'}
    def get_individuals(self):
        individuals_to_images = set(os.listdir(self.image_path))
        individuals_to_labels = set(os.listdir(self.label_path))
        individuals = list(individuals_to_images & individuals_to_labels)
        return individuals

    def load_data(self):
        individual_folders = self.get_individuals()
        image_indiv_path = [os.path.join(self.image_path, i) for i in individual_folders]
        label_indiv_path = [os.path.join(self.label_path, i) for i in individual_folders]
        images = list()
        vec_labels = list()
        labels = list()
        individuals = list()
        emotion_index_map = self.target_emotion_map


        for indiv, image_indiv, label_indiv in zip(individual_folders, image_indiv_path, label_indiv_path):
            image_session = set(os.listdir(image_indiv))
            label_session = set(os.listdir(label_indiv))
            both_session = list(image_session & label_session)


            for one_session in both_session:
                label_indiv_session_txt = glob(os.path.join(label_indiv, one_session, '*.txt'))
                if len(label_indiv_session_txt) == 0:
                    continue
                txt_content = str(int(np.loadtxt(label_indiv_session_txt[0])))
                if self.ck_emotion_map[txt_content] not in self.target_emotion_map.keys():
                    continue
                label_class = self.ck_emotion_map[txt_content]
                images_name_list = glob(os.path.join(image_indiv, one_session, '*.png'))
                images_name_list.sort()
                images.append(self.load_image(images_name_list[0]))
                vec_labels.append(self.ck_emotion_map['0'])
                labels.append(self.target_emotion_map[self.ck_emotion_map['0']])
                individuals.append(indiv)
                for i in range(-3, 0):
                    try:
                        images.append(self.load_image(images_name_list[i]))
                        vec_labels.append(label_class)
                        labels.append(self.target_emotion_map[label_class])
                        individuals.append(indiv)
                    except:
                        print('number of images is less than 4')
        vectorized_labels = self._vectorize_labels(emotion_index_map, vec_labels)
        return (np.array(images, dtype=K.floatx()), 
                np.array(vectorized_labels, dtype=K.floatx()),
                np.array(labels), 
                emotion_index_map)

    def load_image(self, image_path):
        if self.out_channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.detect_face:
            if self.out_channels == 3:
                gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces = self.detector(gray, 1)
                face = max(faces, key=lambda rect: rect.width() * rect.height())
                image = image[face.top():face.bottom(), face.left():face.right()]
   
        if image.shape[0] != self.target_image_size[0] or \
            image.shape[1] != self.target_image_size[1]:
            image = cv2.resize(image, self.target_image_size,
                               interpolation=cv2.INTER_CUBIC)
        return image

    def _vectorize_labels(self, label_index_map, labels):
        label_values = list()
        label_count = len(label_index_map.keys())
        for label in labels:
            label_value = [0] * label_count
            label_value[label_index_map[label]] = 1.0
            label_values.append(label_value)
        return label_values

class NovaEmotions():
    def __init__(self, target_emotion_map, data_path):
        self.target_emotion_map = target_emotion_map
        self.data_path = data_path
    
    def load_data(self):
        subdir = os.listdir(self.data_path)
        images = list()
        labels = list()
        emotion_index_map = self.target_emotion_map
        for emotion in subdir:
            emotion_dir = os.path.join(self.data_path, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            if emotion not in emotion_index_map.keys():
                continue
            images_list = glob(os.path.join(emotion_dir, '*.png'))
            for i in images_list:
                images.append(i)
                labels.append(emotion)
        vectorized_labels = self._vectorize_labels(emotion_index_map, labels)
        return (images, vectorized_labels, emotion_index_map)

    def _vectorize_labels(self, label_index_map, labels):
        label_values = list()
        label_count = len(label_index_map.keys())
        for label in labels:
            label_value = [0] * label_count
            label_value[label_index_map[label]] = 1.0
            label_values.append(label_value)
        return label_values


                


                    

                
                





            

            

                



