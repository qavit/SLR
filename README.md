# slr-nick
手語辨識（Sign Language Recognition, SLR）專案。此專案將使用 Google AI 的 [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) 解決方案分析手語影片（本文件附有[簡介](#mediapipe-簡介)），並使用 [Tensorflow](https://www.tensorflow.org/?hl=zh-tw) [Keras](https://www.tensorflow.org/guide/keras?hl=zh-tw) 訓練 RNN（循環神經網路）模型，最終目標是在 [learnai2024-team3-project](https://github.com/learnai2024-team3-project) 組織的協作下創造一個能辨識台灣手語（Taiwanese Sign Language, TSL）的應用程式。

本專案獲得 [nicknochnack/ActionDetectionforSignLanguage](https://github.com/nicknochnack/ActionDetectionforSignLanguage) 等 GitHub 儲存庫的啟發。


## `requirements.txt`
本專案的相依套件列表存放在 `requirements.txt` 檔案中。

### 記錄相依套件
```sh
pip freeze > requirements_exhaustive.txt
```
手動刪除 `requirements_exhaustive.txt` 中不直接相關的套件。將剩餘的部分另存為 `requirements.txt`

### 安裝相依套件
```sh
pip install -r requirements.txt
```

## `mp.py`
- **警告：此腳本即將棄用！因為保留重要的函式，例如 LSTM 的建構、訓練和評估，所以暫時保留。一部分跟 MediaPipe 的有關的功能暫時轉移到 `fetch_kp_yt.py`**。
- 此腳本（2024-08-02 更新）主要功能是：
   1. 為自定義的語彙標籤集合（例如變數 `new_corpus`）。
   2. 使用函式 `collect_videos` 透過內建的 webcam 錄製各個語彙標籤的手語影片。
   3. 播放手語影片時即時使用 MediaPipe 執行人體姿勢偵測、繪製landmarks，提取關鍵點（keypoints）
   4. 使用 Tensorflow Keras 搭建、訓練和評估 LSTM（長短期記憶模型）。
   5. 訓練完的 LSTM 之後，開啟 webcam，讓使用者比出手勢，程式會即時預測最有可能代表該手勢的標籤，並可視化預測機率。
- 此腳本是參考 [nicknochnack/ActionDetectionforSignLanguage](https://github.com/nicknochnack/ActionDetectionforSignLanguage) 提供的 Jupyter Notebook 所撰寫。可以搭配作者 Nicholas Renotte 的 [YouTube 影片]( https://www.youtube.com/watch?v=doDUihpj6ro)學習。
- 原本各大標題下的程式碼區塊大多被我改寫成函式，為本專案將來的**模組化**（modularization）作準備。
- 根據初步測試，此腳本的模型有過擬合（overfitting）的問題，有必要對模型的設計和超參數（hyperparameters）的調整做進一步的考察。




## `fetch_kp_yt.py`

此腳本（2024-08-04 更新）可以在 Shell 中執行，從 YouTube 下載手語影片，並使用 MediaPipe 進行人體姿勢偵測。

此腳本之後將作為的模組

### 功能概述

1. **影片下載**：根據提供的 CSV 檔案中的 URL，從 YouTube 下載影片。
2. **影片播放**：使用 OpenCV 讀取並播放下載的影片。
3. **姿勢偵測**：在播放影片時，使用 MediaPipe 即時執行人體姿勢偵測。
4. **關鍵點提取**：計算並儲存偵測到的人體關鍵點座標。

### 安裝需求

此腳本的需求為

- numpy
- pandas
- requests
- opencv-python (cv2)
- pytubefix
- Pillow (PIL)
- mediapipe



### 使用方法

1. 準備一個 CSV 檔案，包含要下載的 YouTube 影片 URL 和對應的標籤。
2. 確保 `settings.py` 檔案中的路徑設定正確。
3. 執行腳本，使用不同的命令列參數來執行不同的功能。

### 主要命令列參數

- `-u` 或 `--url_path`：指定 CSV 檔案的路徑（預設為 `settings.py` 中的 `URL_PATH`）
- `-v` 或 `--video_dir`：指定下載影片的儲存目錄（預設為 `settings.py` 中的 `VIDEO_DIR`）
- `-k` 或 `--keypoint_dir`：指定關鍵點資料的儲存目錄（預設為 `settings.py` 中的 `KP_DIR`）
- `-l` 或 `--download`：下載（down**l**oad） CSV 檔案中指定的影片
- `-s` 或 `--show`：顯示（**s**how）已下載的影片清單
- `-d` 或 `--detect`：執行姿勢偵測（**d**etect）並播放結果
- `-e` 或 `--extract`：提取（**e**xtract）關鍵點資料並儲存

### 範例指令

1. 下載影片：
   ```sh
   python video_downloader.py -l
   ```

2. 顯示已下載的影片清單：
   ```sh
   python video_downloader.py -s
   ```

3. 執行姿勢偵測：
   ```sh
   python video_downloader.py -d
   ```

4. 執行姿勢偵測並提取關鍵點：
   ```sh
   python video_downloader.py -e
   ```

### 注意事項

- 請確保您有足夠的網路頻寬和儲存空間來下載和處理影片。
- 姿勢偵測可能需要較高的計算資源，請在適當的硬體上運行。
- 使用 YouTube 影片時，請遵守相關的使用條款和版權規定。

### 進階使用

- 您可以修改 `settings.py` 檔案來自定義各種路徑設定。
- 腳本中的 `draw_styled_landmarks` 函數可以調整以改變繪製的風格。
- `extract_keypoints` 函數提供了關鍵點資料的提取，可以根據需求進行進一步的資料分析。

## MediaPipe 簡介
### Landmark
在 MediaPipe 中 landmark 指的是模型識別和標記的人體或臉部上的關鍵點。這些關鍵點可以用來描述身體或臉部的姿態和位置。以下是一些常見的 MediaPipe landmarks 模型：

1. **Face Mesh**：
    - 用於臉部特徵點檢測，包含 468 個臉部關鍵點。
    - 可以用於臉部動畫、表情識別等應用。

2. **Pose**：
    - 用於全身姿態檢測，包含 33 個身體關鍵點。
    - 可以用於動作識別、運動分析等應用。

3. **Hands**：
    - 用於手勢識別，包含左、右手各 21 個手部關鍵點。
    - 可以用於手勢控制、手語識別等應用。

4. **Holistic**：
    - 結合臉部、姿態和手部模型，提供全面的姿態檢測。
    - 可以同時檢測身體、臉部和手部的關鍵點。

#### Landmark 的結構

每個 landmark 都包含如下資訊：
- `x`：在圖片中的水平坐標（通常在0到1之間）。
- `y`：在圖片中的垂直坐標（通常在0到1之間）。
- `z`：深度坐標（相對於圖片平面，可以是正值或負值）。
- `visibility`（僅限於 pose 模型）：關鍵點的可見性分數，範圍是0到1。
