# Define constants
max_length = 84  # Ensure this matches the training script

def create_features(hand_landmarks, x_, y_):
    """
    Normalizes hand landmark coordinates based on minimum x and y values.

    Parameters:
        hand_landmarks: Mediapipe hand landmarks object.
        x_ (list): List of x coordinates.
        y_ (list): List of y coordinates.

    Returns:
        list: Normalized feature array (data_aux).
    """
    features = []
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        features.append(x - min(x_))
        features.append(y - min(y_))



    # Pad the feature array to match the expected feature length
    if len(features) < max_length:
        features.extend([0] * (max_length - len(features)))

    return features