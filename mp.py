"""
This script builds corpus and trains language models for sign language recognition
This project is dedicated to Taiwanese Sign Languages (TSL).

Usage:
python mp.py <input_file> (future) 

Arguments:
- input_file (str): Path to the input data file. (future)
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

##################################################################
# labels, e.g., ['paper', 'scissors', 'rock'] or [f'number{i}' for i in range(10)]
new_corpus = np.array([f'number{i}' for i in range(10)]) 
print(new_corpus)

MODEL_NAME = 'tsl001'
MODEL_DIR = os.path.join('Models')
DATA_DIR = os.path.join('MP_Data')  # Directory for keypoint data (.npy) acquired by MediaPipe
LOG_DIR = os.path.join('Logs')      # Directory for the log files to be parsed by TensorBoard
N_VIDEOS = 30       # number of videos for each label
N_FRAMES = 30      #  number of frames in each video

N_KEYPTS = 1662       # number of keypoints; 
#          63, if single hand
#        1662, if face + pose + both hands

LOSS = 'categorical_crossentropy'
OPTIM = 'Adam'

##################################################################


def mp_detect(image, mp_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = mp_model.process(image)              # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    # Note that FACE_CONNECTIONS is renamed to FACEMESH_CONTOURS.
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), # landmark_drawing_spec
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1) # connection_drawing_spec
    #                          ) 
    # # Draw pose connections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                          mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
    #                          mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    #                          ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # # Draw right hand connections  
    # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
    #                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
    #                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    #                          ) 


def extract_keypoints(results):
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return lh #np.concatenate([pose, face, lh, rh])


def test_cam(use_mp=False):
    """
    Test the webcam. Press q to close the window.

    Parameters:
    - use_mp (boolean): Use MediaPipe to detect pose and draw the landmarks. Default: False.
    """

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Read feed
        _, frame = cap.read()

        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        if use_mp:
            # Use MediaPipe to detect holistic model 
            image, results = mp_detect(frame, holistic)
            
            # Use MediaPipe to draw landmarks
            draw_styled_landmarks(image, results)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 

    return None



    """
    Creating folder for collecting video frames.
    
    Parameters
    - labels (array_like): 1d array of labels. Each label is a string. For example: `['car', 'shoe', 'lake']`.
    """

    # !? assert *labels is 1d array* 
    # !? assert *each item in labels is a string*

    for label in labels: 
        for idx_video in range(N_VIDEOS):
            try: 
                os.makedirs(os.path.join(DATA_DIR, label, str(idx_video)))
            except:
                pass 

    print(f'{len(labels)} folders created.')


def collect_videos(labels, idx0=0):
    """
    Open built-in webcam, record videos for idividual signs, extract keypoint data, and save as numpy file.

    Parameters:
    - labels (array_like): 1d array of labels. Each label is a string. For example: `['car', 'shoe', 'lake']`.
    - idx0 (int): the index value of the first video in the collection. Default: 0.
    """

    # !? assert *labels is a 1d array* 
    # !? assert *each item in labels is a string*
    assert int(idx0) == idx0

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for label in labels:
            for idx_video in range(idx0, idx0 + N_VIDEOS):
                txt = f'Recording Video No. {idx_video} of Label "{label}"'

                for idx_frame in range(N_FRAMES):

                    # Read feed
                    _, frame = cap.read()

                    # Use MediaPipe to detect holistic model 
                    image, mp_results = mp_detect(frame, holistic)

                    # Use MediaPipe to draw landmarks
                    draw_styled_landmarks(image, mp_results)

                    # Apply wait logic & display text in the window
                    if idx_frame == 0: 
                        cv2.putText(image, 'START RECORDING', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, txt, (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(250)

                    else:
                        countdown = f'{N_FRAMES - idx_frame:02d}'
                        cv2.putText(image, countdown, (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)                     
                        cv2.putText(image, txt, (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    # Extract keypoint data from frames and save data as numpy files
                    keypoints = extract_keypoints(mp_results)
                    npy_path = os.path.join(DATA_DIR, label, str(idx_video), str(idx_frame))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        cv2.destroyAllWindows()


def preprocess(labels):
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
    - labels (array_like): 1d array of labels.

    Returns:
    - train_and_test (tuple): training data and testing data. Each entry is an array.
    """

    # !? assert *labels is a 1d array* 
    # !? assert *each item in labels is a string*

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


##################################################################
# MAIN

## Define MediaPipe Holistic pipeline
mp_holistic = mp.solutions.holistic                 # Holistic model
mp_drawing = mp.solutions.drawing_utils             # Drawing utilities

## Build your corpus, e.g. [number0, number1, ..., number9]
# create_folder(new_corpus)
# collect_videos(new_corpus)

X_train, X_test, y_train, y_test = preprocess(new_corpus)

print(f'{X_train.shape = }') 
print(f'{X_test.shape = }') 
print(f'{y_train.shape = }') 
print(f'{y_test.shape = }')  

#################################################################
# MAIN - MODEL TRAINING

tb_callback = TensorBoard(log_dir=LOG_DIR)

print('\n' + '='*60)
print('Building an LSTM Neural Network...\n')

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,N_KEYPTS)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(new_corpus.shape[0], activation='softmax'))


# Train the model
model.compile(optimizer=OPTIM, loss=LOSS, metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

# Summarize the model training
model.summary()

# Save weights as HDF5 format (or Keras format)
model_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.keras')
model.save(model_path)
# del model

# Reload weights and rebuild the model
model.load_weights(model_path)


##################################################################
# MAIN - MODEL EVALUATION

y_actual = np.argmax(y_test, axis=1).tolist()

y_predicted = model.predict(X_test)
y_predicted = np.argmax(y_predicted, axis=1).tolist()

multilabel_confusion_matrix(y_actual, y_predicted)

accuracy_score(y_actual, y_predicted)
