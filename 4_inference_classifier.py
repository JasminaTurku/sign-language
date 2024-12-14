import pickle
import cv2
import mediapipe as mp
import numpy as np
from utils.draw_hand_landmarks import draw_hand_landmarks
from utils.speech_to_text import speech_to_text_threaded
from utils.text_to_speech import text_to_speech_threaded
import os
from utils.draw_predicted_character import draw_predicted_character
from utils.extract_hand_landmark_coordinates import extract_hand_landmark_coordinates
from utils.create_features import create_features

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Define the data directory
DATA_DIR = './data'

# Get sorted list of directory names
labels = sorted(os.listdir(DATA_DIR))  # Ensure the labels are sorted

# Create label mapping and reverse mapping
label_mapping = {label: idx for idx, label in enumerate(labels)}
reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# -----------------------------------------------------
# UNCOMMENT THE FOLLOWING LINE TO TEST SPEECH TO TEXT
# speech_to_text_threaded()
# -----------------------------------------------------

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            draw_hand_landmarks(frame, hand_landmarks)

            # Extract x and y coordinates
            x_, y_ = extract_hand_landmark_coordinates(hand_landmarks)

            # Use create_features to normalize and prepare the feature array
            features = create_features(hand_landmarks, x_, y_)

            # Make predictions
            prediction = model.predict([np.asarray(features)])
            predicted_character = reverse_label_mapping[int(prediction[0])]

            # Draw the predicted character on the frame
            draw_predicted_character(frame, x_, y_, W, H, predicted_character)

            text_to_speech_threaded(predicted_character)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
