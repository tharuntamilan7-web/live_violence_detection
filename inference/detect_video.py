import cv2
import numpy as np
import tensorflow as tf
import sys

MODEL = "models/violence_model.h5"
SEQ_LEN = 50
IMG_SIZE = 128

video_path = sys.argv[1]
model = tf.keras.models.load_model(MODEL)

cap = cv2.VideoCapture(video_path)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frames.append(resized / 255.0)

    if len(frames) == SEQ_LEN:
        X = np.expand_dims(frames, axis=0)
        pred = model.predict(X, verbose=0)[0]

        label = "VIOLENCE" if np.argmax(pred)==1 else "NORMAL"
        conf = np.max(pred)*100

        cv2.putText(frame, f"{label} {conf:.1f}%",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255) if label=="VIOLENCE" else (0,255,0), 2)

        frames.pop(0)

    cv2.imshow("Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
