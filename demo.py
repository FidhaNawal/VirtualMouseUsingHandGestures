import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np  # Added numpy import for smoothing

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
landmark_drawing_spec = drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=5)
connection_drawing_spec = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2)

screen_width, screen_height = pyautogui.size()
index_y = 0

# Smoothing variables
prev_x, prev_y = 0, 0
smoothing = 5  # Higher number = more smoothing

# Click variables
last_click_time = 0
click_distance = 30  # Distance threshold for click detection

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                                        landmark_drawing_spec, connection_drawing_spec)
            landmarks = hand.landmark
            
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                if id == 8:  # Index finger
                    cv2.circle(img=frame, center=(x, y), radius=30, color=(0, 255, 255), thickness=2)
                    
                    # Calculate raw cursor position
                    raw_x = screen_width / frame_width * x
                    raw_y = screen_height / frame_height * y * 1.5
                    
                    # Apply smoothing
                    if prev_x == 0:
                        # First detection
                        cursor_x, cursor_y = raw_x, raw_y
                    else:
                        # Apply smoothing formula
                        cursor_x = prev_x + (raw_x - prev_x) / smoothing
                        cursor_y = prev_y + (raw_y - prev_y) / smoothing
                    
                    # Update previous positions for next frame
                    prev_x, prev_y = cursor_x, cursor_y
                    
                    # Move cursor to smoothed position
                    pyautogui.moveTo(cursor_x, cursor_y)
                    index_y = cursor_y
                
                if id == 4:  # Thumb
                    cv2.circle(img=frame, center=(x, y), radius=30, color=(0, 255, 255), thickness=2)
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    
                    # Calculate 2D distance between thumb and index finger
                    distance = np.sqrt((thumb_x - prev_x)**2 + (thumb_y - index_y)**2)
                    
                    # Display the distance for debugging
                    cv2.putText(frame, f"Distance: {int(distance)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Click detection with improved logic
                    if distance < click_distance:
                        current_time = time.time()
                        if current_time - last_click_time > 1:  # 1 second delay between clicks
                            pyautogui.click()
                            last_click_time = current_time
                            # Visual feedback for click
                            cv2.putText(frame, "CLICK!", (frame_width // 2 - 50, frame_height // 2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()