from settings import DATA_DIR, N_VIDEOS, N_FRAMES
from fetch_ds import *

import os
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

##################################################################

def preprocess_keypoints(labels):
    """
    Read the keypoint data from numpy files.
    Convert the keypoint data into array (`X`).
    Convert the label number into binary class matrix (`y`).

    Info:
    - `X.ndim = 3`
    - `y.ndim = 2`
    - `X.shape = (len(labels)*N_VIDEOS, N_FRAMES, 1662)`
    - `y.shape = (len(labels)*N_VIDEOS, len(labels))

    Parameters:
    - labels (array): 1d array of labels.

    Returns:
    - train_and_test (tuple): training data and testing data. Each entry is an array.
    """

    label_map = {label:num for num, label in enumerate(labels)}
    video_list, label_num_list = [], []

    for label in labels:
        video_gen = np.array(os.listdir(os.path.join(DATA_DIR, label))).astype(int)
        print(f'{label} : {len(video_gen)} videos')
        
        for idx_video in video_gen:
            video = []

            for idx_frame in range(N_FRAMES):
                keypoints = np.load(os.path.join(DATA_DIR, label, str(idx_video), f'{idx_frame}.npy'))
                video.append(keypoints)
            
            video_list.append(video)
            label_num_list.append(label_map[label])

    # Converts video_list to numpy.ndarray
    X = np.array(video_list)

    # Converts labels to binary class matrix
    y = to_categorical(label_num_list).astype(int)

    train_and_test = train_test_split(X, y, test_size=0.05)

    return train_and_test


def evaluate_model(X_test, y_test):
    """

    Args:
        X_test (array):
        y_test (array):
    """
    y_actual = np.argmax(y_test, axis=1).tolist()
    y_predicted = model.predict(X_test)
    y_predicted = np.argmax(y_predicted, axis=1).tolist()

    multilabel_confusion_matrix(y_actual, y_predicted)
    accuracy_score(y_actual, y_predicted)

##################################################################
# MAIN

# labels, e.g., ['paper', 'scissors', 'rock'] or [f'number{i}' for i in range(10)]
# new_corpus = np.array([f'number{i}' for i in range(10)])
# print(new_corpus)

## Build your corpus, e.g. [number0, number1, ..., number9]
# create_folder(new_corpus)
# collect_videos(new_corpus)

# X_train, X_test, y_train, y_test = preprocess_keypoints(new_corpus)

# print(f'{X_train.shape = }')
# print(f'{X_test.shape = }')
# print(f'{y_train.shape = }')
# print(f'{y_test.shape = }')

#################################################################
# MAIN - MODEL TRAINING

# tb_callback = TensorBoard(log_dir=LOG_DIR)

# print('\n' + '='*60)
# print('Building an LSTM Neural Network...\n')

# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, N_KEYPTS)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(new_corpus.shape[0], activation='softmax'))


# Train the model
# model.compile(optimizer=OPTIM, loss=LOSS, metrics=['categorical_accuracy'])
# model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

# Summarize the model training
# model.summary()

# Save weights as HDF5 format (or Keras format)
# model_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.keras')
# model.save(model_path)
# del model

# Reload weights and rebuild the model
# model.load_weights(model_path)

