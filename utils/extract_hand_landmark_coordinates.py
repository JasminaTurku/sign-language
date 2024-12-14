def extract_hand_landmark_coordinates(hand_landmarks):
    """
    Extracts hand landmark coordinates.

    Parameters:
        hand_landmarks: Mediapipe hand landmarks object.

    Returns:
        tuple: Lists of x and y coordinates.
    """
    x_ = []
    y_ = []
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        x_.append(x)
        y_.append(y)
    return x_, y_