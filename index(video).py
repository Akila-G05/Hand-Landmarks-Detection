import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options) 

cap = cv2.VideoCapture(0) #if you want to use a video file, replace 0 with the filename

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
  'Annotated Image',
  cv2.WND_PROP_FULLSCREEN,
  cv2.WINDOW_GUI_NORMAL
)

while(True):
  ret, frame = cap.read()

  if not ret:
    break

  # Mirror / flip camera horizontally FIRST
  frame = cv2.flip(frame, 1)

  i = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=i)
  result = detector.detect(mpImage)

  annotatedImage = draw_landmarks_on_image(mpImage.numpy_view(), result)
  annotatedImageBGR = cv2.cvtColor(annotatedImage,cv2.COLOR_RGB2BGR)

  cv2.imshow('Annotated Image', annotatedImageBGR)

  if cv2.waitKey(1) == 27:
    break

cap.release()
cv2.destroyAllWindows()