import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

from core.notes import index_to_note
from core.audio import NotePlayer

note_player = NotePlayer()

right_stable_label = "No hand"
right_candidate = "No hand"
right_count = 0

left_stable_label = "No hand"
left_candidate = "No hand"
left_count = 0


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="assets/hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
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

    right_label = "No hand"
    left_label = "No hand"

    if result.hand_landmarks:
        hands = result.hand_landmarks
        for i, hand in enumerate(result.hand_landmarks):
            pts = [(lm.x, lm.y, lm.z) for lm in hand]
            if result.handedness[i][0].display_name == "Right":
                left_label = classify(pts)
            elif result.handedness[i][0].display_name == "Left":
                right_label = classify(pts)

    cv2.putText(frame, "LH: " + left_label + ", RH: " + right_label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # right hand stability
    if right_label == right_candidate:
        right_count += 1
    else:
        right_candidate = right_label
        right_count = 1

    if right_count >= 3:
        right_stable_label = right_candidate
    
    if right_stable_label != "No hand":
        note_player.play_note("right", right_stable_label)
    else:
        note_player.stop_note("right")


    # left hand stability

    if left_label == left_candidate:
        left_count += 1
    else:
        left_candidate = left_label
        left_count = 1

    if left_count >= 3:
        left_stable_label = left_candidate
    
    if left_stable_label != "No hand":
        note_player.play_note("left", left_stable_label)
    else:
        note_player.stop_note("left")

    cv2.imshow("Hand", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
note_player.stop_synth()
