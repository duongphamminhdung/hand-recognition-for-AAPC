from utils import draw_landmarks_on_image
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pynput import keyboard, mouse

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
vid = cv2.VideoCapture(0)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)  

while(True):
    kb = keyboard.Controller()
    Key = keyboard.Key
    ret, frame = vid.read(cv2.CAP_V4L2)
    cv2.imwrite('frame.jpg', frame)
    image = mp.Image.create_from_file('frame.jpg')
    detection_result = detector.detect(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imshow('im', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    if len(detection_result.hand_landmarks) != 0:
        # print(detection_result.hand_landmarks[0])
        if (detection_result.hand_landmarks[0][6].y < detection_result.hand_landmarks[0][8].y) and  detection_result.hand_landmarks[0][9].y < detection_result.hand_landmarks[0][12].y:
            kb.press(Key.up)
            # print('up')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()