import cv2
import csv
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.notes import NOTES, note_to_index

SAMPLES_PER_PRESS = 20
COOLDOWN = 0.5
SAMPLE_INTERVAL = 0.05  # seconds between samples

last_save = 0
current_label = None

# recording state
recording = False
samples_taken = 0
last_sample_time = 0

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="assets/hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

file = open("data/kodaly_dataset.csv", "w", newline="")
writer = csv.writer(file)

header = ["label"] + [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]
writer.writerow(header)


def flatten_landmarks(hand_landmarks):
    return [v for lm in hand_landmarks for v in (lm.x, lm.y, lm.z)]


cap = cv2.VideoCapture(0)

print("Controls: 0-6 select note, SPACE record, ESC exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # label selection
    # -----------------------------
    if ord('0') <= key <= ord('6'):
        current_label = key - ord('0')
        print("Selected:", NOTES[current_label])

    display = NOTES[current_label] if current_label is not None else "NO LABEL"
    cv2.putText(frame, f"Label: {display}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # -----------------------------
    # start recording
    # -----------------------------
    if key == 32 and current_label is not None:
        now = time.time()

        if not recording and (now - last_save > COOLDOWN):
            recording = True
            samples_taken = 0
            print("Recording started...")

    # -----------------------------
    # recording loop (non-blocking)
    # -----------------------------
    if recording:
        now = time.time()

        if result.hand_landmarks:
            if now - last_sample_time >= SAMPLE_INTERVAL:
                hand = result.hand_landmarks[0]
                writer.writerow([current_label] + flatten_landmarks(hand))

                samples_taken += 1
                last_sample_time = now

        # stop condition
        if samples_taken >= SAMPLES_PER_PRESS:
            recording = False
            last_save = now
            print("Recording finished:", NOTES[current_label])

    cv2.imshow("Dataset", frame)

    if key == 27:
        break

file.close()
cap.release()
cv2.destroyAllWindows()
detector.close()