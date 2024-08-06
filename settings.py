import os

LANG = 'zh' # options: ['en', 'zh']

FONT_PATH = os.path.join('Utils', 'NotoSansCJKtc-Regular.otf')
URL_PATH =  os.path.join('URLs', 'urls02.csv')

VIDEO_DIR = 'Videos'
KP_DIR = 'Keypoints'

MODEL_NAME = 'tsl001'
MODEL_DIR = os.path.join('Models')
 
LOG_DIR = os.path.join('Logs')      # Directory for the log files to be parsed by TensorBoard


KP_ZEROS = [33*4, 468*3, 21*3, 21*3] # key point numbers [pose, face, lhand, rhand]

N_VIDEOS = 50      # number of videos for each label
N_FRAMES = 1      # number of frames in each video

N_KEYPTS = 63       # number of keypoints; 
#          63, if single hand
#        1662, if face + pose + both hands

LOSS = 'categorical_crossentropy'
OPTIM = 'Adam'