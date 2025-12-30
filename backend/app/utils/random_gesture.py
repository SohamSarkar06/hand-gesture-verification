import random

# Must MATCH dataset folder names exactly
GESTURES = [
    "palm",         # open hand
    "fist",
    "thumb",        # thumbs up
    "ok",           # OK sign
    "index",        # index finger
    "c",            # C shape
    "down",         # hand down
    "l"             # L shape
]

def get_random_gesture():
    return random.choice(GESTURES)
