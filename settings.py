import os

FONT_PATH = os.path.join('Utils', 'NotoSansCJKtc-Regular.otf')
URL_PATH =  os.path.join('URLs', 'urls02.csv')

VIDEO_DIR = 'Videos'
KP_DIR = 'Keypoints'

MODEL_NAME = 'tsl001'
MODEL_DIR = os.path.join('Models')

KP2_DIR = os.path.join('MP_Data')  # Directory for keypoint data (.npy) acquired by MediaPipe
LOG_DIR = os.path.join('Logs')      # Directory for the log files to be parsed by TensorBoard


KP_ZEROS = [33*4, 468*3, 21*3, 21*3] # key point numbers [pose, face, lhand, rhand]

N_VIDEOS = 30       # number of videos for each label
N_FRAMES = 30      # number of frames in each video

N_KEYPTS = 1662       # number of keypoints; 
#          63, if single hand
#        1662, if face + pose + both hands

LOSS = 'categorical_crossentropy'
OPTIM = 'Adam'