import cv2

def draw_predicted_character(frame, xCords, yCords, W, H, predicted_character):
    # Calculate bounding box coordinates
    x1 = int(min(xCords) * W) - 10
    y1 = int(min(yCords) * H) - 10
    x2 = int(max(xCords) * W) - 10

    # Center the text horizontally
    center_x = (x1 + x2) // 2

    # Get text size for alignment
    text_width, text_height = cv2.getTextSize(predicted_character, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
    text_x = center_x - text_width // 2  # Horizontally center the text
    text_y = max(y1 - 10, text_height)   # Place text above bounding box, ensure visibility

    # Draw the label
    cv2.putText(frame, predicted_character, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
