import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def verify_ok(lm):
    return dist(lm[4], lm[8]) < 0.05  # thumb-index close

def verify_l(lm):
    return dist(lm[4], lm[8]) > 0.15  # thumb-index far

def verify_palm(lm):
    tips = [8, 12, 16, 20]
    wrist = lm[0]
    return all(dist(lm[t], wrist) > 0.2 for t in tips)

GESTURE_CHECKS = {
    "ok": verify_ok,
    "l": verify_l,
    "palm": verify_palm,
}

def verify_by_landmarks(image_path, expected):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return False

    lm = result.multi_hand_landmarks[0].landmark
    checker = GESTURE_CHECKS.get(expected)

    if not checker:
        return True  # fallback to CNN only

    return checker(lm)
