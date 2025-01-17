import time
import ctypes
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize volume adjustment
last_change = time.time()

def change_volume(direction):
    """Adjust system volume on Windows using ctypes."""
    VK_VOLUME_UP = 0xAF
    VK_VOLUME_DOWN = 0xAE
    if direction == 'up':
        ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 2, 0)  # Key up
    elif direction == 'down':
        ctypes.windll.user32.keybd_event(VK_VOLUME_DOWN, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_VOLUME_DOWN, 0, 2, 0)  # Key up

# Video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame.")
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lmList = []
                h, w, c = img.shape
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                # Calculate distance between index fingertip (id=8) and thumb tip (id=4)
                tipId, thumbTipId = 8, 4
                tipX, tipY = lmList[tipId][1], lmList[tipId][2]
                thumbTipX, thumbTipY = lmList[thumbTipId][1], lmList[thumbTipId][2]
                distance = ((tipX - thumbTipX)**2 + (tipY - thumbTipY)**2)**0.5
                print(f"Distance: {distance}")

                # Adjust volume based on distance with cooldown
                if distance < 70 and time.time() - last_change > 0.5:
                    print("Volume Down Triggered")
                    change_volume('down')
                    last_change = time.time()
                elif distance > 130 and time.time() - last_change > 0.5:
                    print("Volume Up Triggered")
                    change_volume('up')
                    last_change = time.time()

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

