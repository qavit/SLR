import numpy as np
import pandas as pd
import requests
import cv2
import argparse
import os
import re
from pytubefix import YouTube 
from PIL import ImageFont, ImageDraw, Image
from settings import FONT_PATH, URL_PATH, VIDEO_DIR, KP_DIR, KP_ZEROS

def read_csv(file_path):
    """Read a CSV file and extract labels and URLs.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: Two lists, one containing labels and the other containing URLs.
    """
    df = pd.read_csv(file_path)
    labels = df['Label'].tolist()
    urls = df['URL'].tolist()
    return labels, urls

def download_YouTube(url, file_path):
    """Download a video from the given YouTube URL and save it to a file.
    
    This downloader is based on `pytubefix`. See more at https://pytubefix.readthedocs.io/en/latest/.

    Args:
        url (str): The URL of the video to download.
        file_path (str): The path where the video will be saved.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path=file_path.rsplit('/', 1)[0], filename=file_path.rsplit('/', 1)[1])
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def download_with_requests(url, file_path):
    """Download a file from a given URL and save it to a specified file path.

    This function uses the `requests` library to download a file in chunks from the provided URL.
    The file is saved to the specified file path. It ensures the file is not empty before returning success.

    Args:
        url (str): The URL of the file to download.
        file_path (str): The path where the downloaded file will be saved.

    Returns:
        bool
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        if os.path.getsize(file_path) > 0:  # Ensure the file is not empty
            return True
    return False

def draw_styled_landmarks(image, results, drawing, model):
    """Draw the landmarks on the image.

    Note that `FACE_CONNECTIONS` is renamed to `FACEMESH_CONTOURS`.

    Args:
        image (ndarray): The image on which to draw the landmarks.
        results (object): The results from the holistic model containing the landmarks.
        drawing (object): The MediaPipe solution drawing utilities.
        model (object): The MediaPipe holistic model.
    
    Returns:
        None
    """

    draw = drawing.draw_landmarks
    spec = drawing.DrawingSpec

    # Draw face landmarks
    draw(image, results.face_landmarks, model.FACEMESH_CONTOURS,
        spec(color=(80, 110, 10), thickness=1, circle_radius=1),  # landmark_drawing_spec
        spec(color=(80, 256, 121), thickness=1, circle_radius=1)  # connection_drawing_spec
        )

    # Draw pose landmarks
    draw(image, results.pose_landmarks, model.POSE_CONNECTIONS,
        spec(color=(80, 22, 10), thickness=2, circle_radius=4),
        spec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

    # Draw left hand landmarks
    draw(image, results.left_hand_landmarks, model.HAND_CONNECTIONS,
        spec(color=(121, 22, 76), thickness=2, circle_radius=4),
        spec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

    # Draw right hand landmarks
    draw(image, results.right_hand_landmarks, model.HAND_CONNECTIONS,
        spec(color=(245, 117, 66), thickness=2, circle_radius=4),
        spec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    
    return None

def detect_landmarks(image, model):
    """Detect holistic landmarks in the image.

    Args:
        image (ndarray): The image in which to detect the landmarks.
        model (object): The MediaPipe holistic model.

    Returns:
        tuple: The processed image and the results from the holistic model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert color from BGR to RGB
    image.flags.writeable = False                  # Make image unwriteable
    results = model.process(image)                 # Process image with MediaPipe's model
    image.flags.writeable = True                   # Make image unwriteable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert color from RGB to BGR
    return image, results

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results and create a 1D NumPy array.

    This function extracts keypoint data from the landmarks detected by MediaPipe's holistic model 
    (pose, face, left hand, right hand). For pose landmarks, it includes visibility information. 
    If a set of landmarks is not detected, a zero array of the specified length is used instead. 
    The resulting keypoints are concatenated into a single 1D NumPy array.

    Args:
        results (object): The MediaPipe results object containing the detected landmarks.

    Returns:
        ndarray: A 1D NumPy array containing the concatenated keypoints.
    """
    landmarks = [results.pose_landmarks,
                 results.face_landmarks,
                 results.left_hand_landmarks,
                 results.right_hand_landmarks]

    zero_array = [np.zeros(z) for z in KP_ZEROS]

    keypoints = []
    for i, lm in enumerate(landmarks):
        if lm:
            if i == 0:      # i.e. pose_landmarks
                kp = [[pt.x, pt.y, pt.z, pt.visibility] for pt in lm.landmark]
            else:           # i.e. other landmarks
                kp = [[pt.x, pt.y, pt.z] for pt in lm.landmark]
            kp = np.array(kp).flatten() # Flatten to 1D NumPy array
        else:
            kp = zero_array[i]
        keypoints.append(kp)
    
    return np.concatenate(keypoints)

def put_text_CJK(img, txt, font_path=FONT_PATH, 
                 position=(0, 0), font_size=50, color=(255, 255, 255)):
    """Add CJK (Chinese, Japanese, Korean) text to an image using PIL (Python Imaging Library).

    This function uses the PIL library to add CJK text to an image, which can then be displayed with OpenCV.

    Args:
        img (ndarray): The input image as a numpy array.
        txt (str): The text to be added to the image.
        font_path (str): The path to the TrueType font (.ttf) file. Deafults to FONT_PATH.
        position (tuple, optional): The position (x, y) where the text should be added. Defaults to (0, 0).
        font_size (int, optional): The size of the font. Defaults to 50.
        color (tuple, optional): The color of the text in (R, G, B). Defaults to (255, 255, 255).

    Returns:
        ndarray: The image with the added text as a numpy array.
    """
    font = ImageFont.truetype(font_path, font_size)  # Set the font and text size
    img_pil = Image.fromarray(img)                   # Convert the NumPy array to a PIL image
    draw = ImageDraw.Draw(img_pil)                   # Prepare to draw on the image
    draw.text(position, txt, fill=color, font=font)  # Draw the text on the image
    return np.array(img_pil)                         # Convert the PIL image back to a NumPy array

def play_and_detect(label, video_path, keypoint_dir, extract=False):
    """Play a video, detect holistic landmarks using MediaPipe, and optionally extract keypoints.

    This function plays a video from the given file path, detects holistic landmarks in each frame 
    using MediaPipe, and draws the landmarks on the frame. The video is displayed with the given label 
    added as text. If the `extract` parameter is set to True, keypoints are extracted from the frames 
    and saved as numpy files.

    Args:
        label (str): The label to display on the video.
        video_path (str): The path to load the video file.
        keypoint_dir (str): The directory to save the numpy file for keypoint data.
        extract (bool, optional): Flag indicating whether to extract and save keypoints. Defaults to False.

    Returns:
        None
    """

    # Initialize MediaPipe holistic model and drawing utilities
    import mediapipe as mp
    model = mp.solutions.holistic  # Holistic model
    drawing = mp.solutions.drawing_utils  # Drawing utilities  
   
    with model.Holistic(min_detection_confidence=0.5, 
                              min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: # if frame is read correctly, then ret is True
                break
            
            # Detect and draw the holistic landmarks 
            frame, results = detect_landmarks(frame, holistic)
            draw_styled_landmarks(frame, results, drawing, model)
            
            # Add a text label (CJK characters allowed) to the frame and display both of them.
            frame_with_label = put_text_CJK(frame, label)
            cv2.imshow(f'Video {label}', frame_with_label)

            # Extract keypoint data from frames and save data as numpy files.
            if extract:
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(keypoint_dir, label)
                np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def show_video_list(video_dir, printed=False):
    """Scan directory for 'video_<label>.mp4' files and return the video list as a DataFrame. 
    
    Given the option `printed=True`, the DataFrame is also printed to the console for immediate viewing.

    Args:
        video_dir (str): Directory path containing video files.

    Returns:
        pandas.DataFrame: Columns 'label' and 'video_path' for matching files.
    """
    video_files = []

    # Iterate through all files in the specified directory
    for filename in os.listdir(video_dir):
        # Use regular expression to match filenames in the required format
        match = re.match(r'video_(.+)\.mp4', filename)
        if match:
            video_path = os.path.join(video_dir, filename)
            label = match.group(1)
            video_files.append((label, video_path))

    # Create a pandas DataFrame from the video_files list
    df = pd.DataFrame(video_files, columns=['label', 'video_path'])

    # Print results for confirmation
    if printed == True:
        print(df)

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and play videos from URLs in a CSV file.")
    
    parser.add_argument('--url_path', '-u', type=str, default=URL_PATH,
                        help=f"The path to the CSV file containing video URLs and labels. Default is {URL_PATH}.")
    parser.add_argument('--video_dir', '-v', type=str, default=VIDEO_DIR,
                        help=f"The folder where videos will be downloaded. Default is {VIDEO_DIR}.")
    parser.add_argument('--keypoint_dir', '-k', type=str, default=KP_DIR,
                        help=f"The folder where keypoint data will be saved. Default is {KP_DIR}.")
    parser.add_argument('-l', '--download', action='store_true', 
                        help="Download the videos according to the URLs in url_path.")
    parser.add_argument('-s', '--show', action='store_true', 
                        help="Show the list of downloaded videos.")
    parser.add_argument('-d', '--detect', action='store_true', 
                        help="Detect the pose in the video and display it.")
    parser.add_argument('-e', '--extract', action='store_true', 
                        help="Extract the keypoint data from the video and save it.")
    
    args = parser.parse_args()

    # Print settings
    print(f"Font path: {FONT_PATH}")
    print(f"URL file: {URL_PATH}")
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Keypoints directory: {KP_DIR}")

    # Create the download folders if they don't exist
    for folder in [args.video_dir, args.keypoint_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if args.download:
        # Read the labels and URLs from the CSV file
        labels, urls = read_csv(args.url_path)
        print('Importing:', labels)

        # Process each label and URL pair
        for i, (label, url) in enumerate(zip(labels, urls)):
            video_path = os.path.join(args.video_dir, f"video_{label}.mp4")
            
            # Download the video from YouTube
            if download_YouTube(url, video_path):
                print(f"Video downloaded: {video_path}")
            else:
                print(f"Failed to download video: {url}")

    # Scan for videos in the directory and print the info.
    if args.show:
        show_video_list(args.video_dir, printed=True)
        
    # Detect the landmarks for each video; also extract the keypoints if args.extract is True
    if args.detect or args.extract:
        video_df = show_video_list(args.video_dir)
        for index, row in video_df.iterrows():
            play_and_detect(row['label'], row['video_path'], args.keypoint_dir, extract=args.extract)

