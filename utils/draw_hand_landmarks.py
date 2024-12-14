import mediapipe as mp

def draw_hand_landmarks(frame, hand_landmarks):
    """
    Draws hand landmarks on the given frame.

    Parameters:
        frame (np.array): The video frame.
        hand_landmarks: Mediapipe hand landmarks object.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
