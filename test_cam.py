import cv2
from fetch import detect_landmarks, draw_styled_landmarks

leave_txt = "Press 'q' to close the window."

##################################################################

def test_cam_with_opencv():
    """Test the webcam using OpenCV.

    Press 'q' to close the window.

    Returns:
        None
    """
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        cv2.putText(frame, leave_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def test_cam_with_mediapipe():
    """Test the webcam using OpenCV and MediaPipe.

    Press 'q' to close the window.

    Returns:
        None
    """
    import mediapipe as mp

    # Initialize MediaPipe holistic model and drawing utilities
    model = mp.solutions.holistic
    drawing = mp.solutions.drawing_utils

    with model.Holistic(min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame, results = detect_landmarks(frame, holistic)
            draw_styled_landmarks(frame, results, drawing, model)

            cv2.putText(frame, leave_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

##################################################################

if __name__ == '__main__':
    # test_cam_with_opencv()
    test_cam_with_mediapipe()