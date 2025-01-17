import os
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import pyaudio
import time

# Suppress TensorFlow Lite warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize PyAudio
p = pyaudio.PyAudio()
volume = 0.5  # Initial volume
last_change = time.time()  # Cooldown timer

def callback(in_data, frame_count, time_info, status):
    global volume
    if in_data is None:  # Handle None input
        return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)
    array = np.frombuffer(in_data, dtype=np.float32)
    array *= volume
    return array.tobytes(), pyaudio.paContinue

try:
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    output=True,
                    stream_callback=callback)
    stream.start_stream()
except Exception as e:
    print(f"Audio stream error: {e}")
    exit()

# Video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use backend if needed
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

print("Press 'q' to quit.")

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame. Exiting.")
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # Hand landmark detection
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                # Debugging landmarks
                print("Landmarks:", lmList)

                # Calculate distance between index fingertip and thumb tip
                tipId, thumbTipId = 8, 4
                tipX, tipY = lmList[tipId][1], lmList[tipId][2]
                thumbTipX, thumbTipY = lmList[thumbTipId][1], lmList[thumbTipId][2]
                distance = ((tipX - thumbTipX)**2 + (tipY - thumbTipY)**2)**0.5
                print(f"Distance: {distance}")

                # Adjust volume based on distance with cooldown
                if distance < 70 and time.time() - last_change > 0.5:  # Adjust lower threshold
                    volume = max(0, volume - 0.05)
                    pyautogui.press('volume_down')
                    print(f"Volume Decreased to: {volume:.2f}")
                    last_change = time.time()
                elif distance > 130 and time.time() - last_change > 0.5:  # Adjust upper threshold
                    volume = min(1, volume + 0.05)
                    pyautogui.press('volume_up')
                    print(f"Volume Increased to: {volume:.2f}")
                    last_change = time.time()

                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    p.terminate()
