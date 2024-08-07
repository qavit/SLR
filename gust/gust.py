import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


imgdir = os.path.join('..', 'Images')
imgpaths = [os.path.join(imgdir, img) for img in os.listdir(imgdir)]


######################
# 載入權重，並重新建立模型（僅儲存權重）
model = create_model()  # 創建與儲存時相同的模型架構
model.load_weights('model.h5')

# predictions = model.predict(