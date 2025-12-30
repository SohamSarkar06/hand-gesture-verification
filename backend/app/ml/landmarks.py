import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands.Hands(static_image_mode=True)

def extract_landmarks(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = mp_hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None

    landmarks = []
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks)
