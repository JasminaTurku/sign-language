import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

from utils.create_features import create_features
from utils.extract_hand_landmark_coordinates import extract_hand_landmark_coordinates

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        features = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates
                x_, y_ = extract_hand_landmark_coordinates(hand_landmarks)

                # Use create_features to normalize and prepare the feature array
                features = create_features(hand_landmarks, x_, y_)

            data.append(features)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()