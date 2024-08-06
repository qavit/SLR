import os
import cv2
import numpy as np
from fetch import *
from settings import LANG, N_VIDEOS, N_FRAMES, KP_DIR

##################################################################

def set_up_folder(labels):
    idx0_list = []
    for label in labels: 
        create_folder(os.path.join(KP_DIR, label))
        listdir = os.listdir(os.path.join(KP_DIR, label))
        if len(listdir) == 0:
            idx0 = 0
        else:
            idx0 = np.max(np.array(os.listdir(os.path.join(KP_DIR, label))).astype(int))
        
        for idx_video in range(N_VIDEOS):
            try: 
                os.makedirs(os.path.join(KP_DIR, label, str(idx0 + idx_video)))
            except:
                pass

        idx0_list.append(idx0)

    return dict(zip(labels, idx0_list))

def record_my_videos(labels):
    """Open built-in webcam, record videos for idividual signs, extract keypoint data, and save as numpy file.

    Parameters:
        labels (array_like): 1d array of labels. Each label is a string. For example: `['car', 'shoe', 'lake']`.
    
    Returns:
        None
    """
    import mediapipe as mp
    model = mp.solutions.holistic
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    with model.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        
        for i, label in enumerate(labels):
            video_list = []

            ready_txt = {'en': f'ABOUT TO RECORD {label}, COUNTDOWN: ', 
                         'zh': f'即將開始錄製標籤「{label}」，倒數'}

            for c in [3,2,1]:
                _, frame = cap.read()
                imagec = put_text_CJK(frame, ready_txt[LANG] + str(c),
                                      color=(255,0,255)) 
                cv2.imshow('OpenCV Feed', imagec)
                cv2.waitKey(1000)

            for idx_video in range(N_VIDEOS):
                video = []

                start_txt = {'en': 'START', 'zh': '開始'}
                recording_txt = {'en': f'Recording Video No. {idx_video} of Label "{label}"',
                                 'zh': f'正在錄製標籤「{label}」的第 {idx_video} 部影片'}

                for idx_frame in range(N_FRAMES):
                    _, frame = cap.read()

                    image, results = detect_landmarks(frame, holistic)
                    draw_styled_landmarks(image, results, drawing, model)

                    # Apply wait logic & display text in the window
                    if idx_frame == 0:
                        image = put_text_CJK(image, start_txt[LANG], font_size=50,
                                             color=(0,255,255))
                        image = put_text_CJK(image, recording_txt[LANG], 
                                             position = (350,0), font_size=40)
                        cv2.imshow('OpenCV Feed', image)
                    else:
                        frame_countdown = f'{N_FRAMES - idx_frame:02d}/{N_FRAMES:02d}' 
                        image = put_text_CJK(image, frame_countdown, font_size=50,
                                             color=(255,255,0))                  
                        image = put_text_CJK(image, recording_txt[LANG], 
                                             position = (350,0), font_size=40)
                        cv2.imshow('OpenCV Feed', image)

                    # Extract keypoint data from frames and save data as numpy files
                    keypoints = extract_keypoints(results)
                    #npy_path = os.path.join(KP_DIR, label, str(idx_video), str(idx_frame))
                    #np.save(npy_path, keypoints)
                    video.append(keypoints)

                    print(i, idx_video, idx_frame, keypoints.shape)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                
                video_list.append(video)
            
            X = np.array(video_list)
            #print(X.shape, '\n')
            np.save(os.path.join(KP_DIR, label), X)

        cap.release()
        cv2.destroyAllWindows()

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

    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical

    # Converts labels to binary class matrix
    y = to_categorical(label_num_list).astype(int)

    train_and_test = train_test_split(X, y, test_size=0.05)

    return train_and_test

def clear_data():
    pass

##################################################################
# MAIN FUNCTION


if __name__ == '__main__':
    N = int(input('Enter a number (1-10)'))
    my_labels = np.array([f'number{i}' for i in range(N)])
    print(my_labels)

    # Setup Folders for Collection
    # set_up_folder(my_labels)
    print('Preparing...')
    record_my_videos(my_labels)

    # preprocess_keypoints(my_labels)
