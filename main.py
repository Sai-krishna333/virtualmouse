import cv2
import mediapipe as mp
import pyautogui
import time
import math
import os
import numpy as np
import speech_recognition as sr 
import threading

# Create screenshot folder if not exists
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Initialize MediaPipe Hands (supporting two hands now)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

# Click cooldown
prev_click_time = 0
click_delay = 1

# Gesture control toggle
paused = False

# Voice control flag
voice_command = ""

# Function to calculate distance
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Function to calculate angle between wrist and index
def hand_angle(lm):
    x1, y1 = lm[0]  # wrist
    x2, y2 = lm[8]  # index tip
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

# Voice command handler
def listen_for_voice_commands():
    global paused
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
    while True:
        with mic as source:
            print("Listening for 'pause' or 'resume'...")
            audio = r.listen(source)
            try:
                command = r.recognize_google(audio).lower()
                if "pause" in command:
                    paused = True
                    print("Voice: Gesture Control Paused")
                elif "resume" in command:
                    paused = False
                    print("Voice: Gesture Control Resumed")
            except sr.UnknownValueError:
                pass

# Start voice control in background
threading.Thread(target=listen_for_voice_commands, daemon=True).start()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    hand_positions = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_positions.append(lm_list)

            # Single-hand gestures
            if len(hand_positions) == 1 and not paused:
                index = lm_list[8]
                thumb = lm_list[4]
                middle = lm_list[12]
                ring = lm_list[16]
                pinky = lm_list[20]

                # Move mouse
                pyautogui.moveTo(screen_width * index[0] / w, screen_height * index[1] / h)
                cv2.circle(img, index, 15, (255, 255, 0), -1)  # visual cursor

                if distance(index, thumb) < 40:
                    if time.time() - prev_click_time > click_delay:
                        pyautogui.click()
                        print("Left Click")
                        prev_click_time = time.time()

                elif distance(index, middle) < 40:
                    if time.time() - prev_click_time > click_delay:
                        pyautogui.rightClick()
                        print("Right Click")
                        prev_click_time = time.time()

                elif distance(index, ring) < 40:
                    if time.time() - prev_click_time > click_delay:
                        pyautogui.doubleClick()
                        print("Double Click")
                        prev_click_time = time.time()

                elif distance(thumb, pinky) < 40:
                    if time.time() - prev_click_time > click_delay:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        file_name = f"screenshots/screenshot_{timestamp}.png"
                        pyautogui.screenshot(file_name)
                        print(f"Screenshot saved: {file_name}")
                        prev_click_time = time.time()

                elif distance(index, middle) > 100:
                    pyautogui.scroll(20)  # scroll up
                    print("Scroll Up")

                elif all(distance(index, f) < 40 for f in [thumb, middle, ring, pinky]):
                    paused = True
                    print("Gesture Control Paused")

                # Volume control
                ang = hand_angle(lm_list)
                if -60 < ang < -20:
                    pyautogui.press("volumeup")
                    print("Volume Up")
                elif 20 < ang < 60:
                    pyautogui.press("volumedown")
                    print("Volume Down")

            # Two-hand zoom gesture
            elif len(hand_positions) == 2 and not paused:
                hand1 = hand_positions[0]
                hand2 = hand_positions[1]

                h1_thumb = hand1[4]
                h2_thumb = hand2[4]

                dist = distance(h1_thumb, h2_thumb)

                if dist > 250:
                    pyautogui.hotkey('ctrl', '+')
                    print("Zoom In")
                elif dist < 150:
                    pyautogui.hotkey('ctrl', '-')
                    print("Zoom Out")

    # Status text
    cv2.putText(img, f"{'Paused' if paused else 'Active'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if paused else (0, 255, 0), 2)

    cv2.imshow("AI Virtual Mouse Pro", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
#saikrishna