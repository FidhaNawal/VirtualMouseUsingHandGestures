import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
smoothing = 5
dead_zone = 4

gesture_w, gesture_h = 300, 200
click_cooldown = 1
pinch_start_time = None
last_click_time = 0
scroll_mode = False
scroll_base_y = None

frame_count = 0  # To control processing frequency

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_fingers_up(lm):
    return [
        lm[8].y < lm[6].y,  # Index
        lm[12].y < lm[10].y  # Middle
    ]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rx = w // 2 - gesture_w // 2
    ry = h // 2 - gesture_h // 2
    cv2.rectangle(frame, (rx, ry), (rx + gesture_w, ry + gesture_h), (100, 255, 100), 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Only process every 2nd frame for performance
    frame_count += 1
    if frame_count % 2 != 0:
        cv2.imshow("Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark
            index = lm[8]
            thumb = lm[4]
            middle = lm[12]
            index_base = lm[5]

            ix, iy = int(index.x * w), int(index.y * h)
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            mx, my = int(middle.x * w), int(middle.y * h)

            fx, fy = np.clip(ix - rx, 0, gesture_w), np.clip(iy - ry, 0, gesture_h)
            sx = screen_w * fx / gesture_w
            sy = screen_h * fy / gesture_h
            move_dist = np.hypot(sx - prev_x, sy - prev_y)

            if move_dist > dead_zone and not scroll_mode:
                curr_x = prev_x + (sx - prev_x) / smoothing
                curr_y = prev_y + (sy - prev_y) / smoothing
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

            # Gesture detection
            pinch_dist = get_distance([index.x, index.y], [thumb.x, thumb.y])
            finger_len = get_distance([index.x, index.y], [index_base.x, index_base.y])
            norm_pinch = pinch_dist / finger_len if finger_len > 0 else 1

            fingers = get_fingers_up(lm)
            index_up, middle_up = fingers

            if norm_pinch < 0.5 and not scroll_mode:
                if pinch_start_time is None:
                    pinch_start_time = time.time()
                elif time.time() - pinch_start_time > 0.5:
                    pyautogui.doubleClick()
                    pinch_start_time = None
                    cv2.putText(frame, "DOUBLE CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                if pinch_start_time:
                    if time.time() - pinch_start_time < 0.5:
                        now = time.time()
                        if now - last_click_time > click_cooldown:
                            pyautogui.click()
                            last_click_time = now
                            cv2.putText(frame, "CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    pinch_start_time = None

            # Scroll
            if index_up and middle_up:
                if not scroll_mode:
                    scroll_mode = True
                    scroll_base_y = my
                else:
                    dy = my - scroll_base_y
                    if abs(dy) > 4:
                        pyautogui.scroll(-int(dy / 3))
                        scroll_base_y = my
                    cv2.putText(frame, "SCROLL MODE", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            else:
                scroll_mode = False

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
