import cv
import mediapipe as mp
import pyautogui
import time
cap=cv2.VideoCapture(0)
hand_detector=mp.solutions.hands.Hands()
drawing_utils=mp.solutions.drawing_utils
landmark_drawing_spec = drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=5)
connection_drawing_spec = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2)
screen_width,screen_height=pyautogui.size()
index_y=0
last_click_time = 0
while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    frame_height,frame_width,_=frame.shape
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output=hand_detector.process(rgb_frame)
    hands=output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame,hand,mp.solutions.hands.HAND_CONNECTIONS,landmark_drawing_spec,connection_drawing_spec)
            landmarks=hand.landmark
            for id,landmark in enumerate(landmarks):
                x=int(landmark.x*frame_width)
                y=int(landmark.y*frame_height)
                print(x,y)
                if id==8:
                    cv2.circle(img=frame,center=(x,y),radius=30,color=(0,255,255),thickness=2)
                    index_x=screen_width/frame_width * x 
                    index_y=screen_height/frame_height * y * 1.5
                    pyautogui.moveTo(index_x,index_y)
                if id==4:
                    cv2.circle(img=frame,center=(x,y),radius=30,color=(0,255,255),thickness=2)
                    thumb_x=screen_width/frame_width * x
                    thumb_y=screen_height/frame_height * y
                    print(f"Index Y: {index_y}, Thumb Y: {thumb_y}, Distance: {abs(index_y - thumb_y)}")
                    if abs(index_y - thumb_y)<20:
                         current_time = time.time()
                         if current_time - last_click_time > 1:  # 1 second delay between clicks
                            pyautogui.click()
                            last_click_time = current_time
            
    cv2.imshow("virtual Mouse",frame)
    cv2.waitKey(1)



