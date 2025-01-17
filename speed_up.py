import time
import ctypes
import cv2
import mediapipe as mp
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

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
cap.set(3, 320)  # Set width (lower resolution)
cap.set(4, 240)  # Set height

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

frame_count = 0  # For frame skipping

def volume_control_thread(distance, cooldown):
    """Run volume control in a separate thread."""
    global last_change
    if distance < 50 and time.time() - last_change > cooldown:
        print("Volume Down Triggered")
        change_volume('down')
        last_change = time.time()
    elif distance > 120 and time.time() - last_change > cooldown:
        print("Volume Up Triggered")
        change_volume('up')
        last_change = time.time()

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame.")
            break

        frame_count += 1

        # Process only every 2nd frame
        if frame_count % 2 != 0:
            cv2.imshow("Hand Tracker", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Convert the image to RGB for MediaPipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # Process hand landmarks
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Draw landmarks every 2nd frame
                if frame_count % 4 == 0:
                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

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

                # Start volume adjustment in a separate thread
                thread = threading.Thread(target=volume_control_thread, args=(distance, 0.3))
                thread.start()

        # Show video feed
        cv2.imshow("Hand Tracker", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
