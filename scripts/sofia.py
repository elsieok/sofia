import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

from core.notes import index_to_note
from core.audio import NotePlayer

note_player = NotePlayer()
stable_label = "No hand"
candidate = "No hand"
count = 0


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

cap = cv2.VideoCapture(0)

model = load_model("model/kodaly_model.h5")

def classify(landmarks):
    sample = []
    for x, y, z in landmarks:
        sample.extend([x, y, z])

    sample = np.array(sample, dtype=np.float32).reshape(1, 63)

    pred = model.predict(sample, verbose=0)
    index = np.argmax(pred)

    return index_to_note[index]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    label = "No hand"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        pts = [(lm.x, lm.y, lm.z) for lm in hand]
        label = classify(pts)

    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if label == candidate:
        count += 1
    else:
        candidate = label
        count = 1

    if count >= 3:
        stable_label = candidate
    
    if stable_label != "No hand":
        note_player.play_note(stable_label)
    else:
        note_player.stop_note()

    cv2.imshow("Hand", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
note_player.stop_synth()
