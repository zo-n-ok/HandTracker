import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Set up PyAudio
import pyaudio
import numpy as np

p = pyaudio.PyAudio()
volume = 0.5  # Initial volume

def callback(in_data, frame_count, time_info, status):
    global volume
    array = np.frombuffer(in_data, dtype=np.float32)
    array *= volume
    return array.tobytes(), pyaudio.paContinue

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                output=True,
                stream_callback=callback)
stream.start_stream()

# Video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
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

            # Calculate distance between index fingertip and thumb tip
            tipId = 8  # Index fingertip
            thumbTipId = 4  # Thumb tip
            tipX, tipY = lmList[tipId][1], lmList[tipId][2]
            thumbTipX, thumbTipY = lmList[thumbTipId][1], lmList[thumbTipId][2]
            distance = ((tipX - thumbTipX)**2 + (tipY - thumbTipY)**2)**0.5

            # Adjust volume based on distance
            if distance < 30:  # Adjust threshold as needed
                volume = max(0, min(1, volume - 0.05))
                pyautogui.press('volume_down')
            elif distance > 60:  # Adjust threshold as needed
                volume = min(1, max(0, volume + 0.05))
                pyautogui.press('volume_up')

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()