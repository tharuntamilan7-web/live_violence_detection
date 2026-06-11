import cv2
import numpy as np
import tensorflow as tf
import sys
from alerts.telegram_alert import send_telegram_alert

MODEL_PATH = "models/violence_model.h5"
SEQ_LEN = 50
IMG_SIZE = 128

RTSP_URL = sys.argv[1]   # passed from Flask

model = tf.keras.models.load_model(MODEL_PATH)
cap = cv2.VideoCapture(RTSP_URL)

frames = []
alert_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frames.append(resized / 255.0)

    if len(frames) == SEQ_LEN:
        X = np.expand_dims(frames, axis=0)
        pred = model.predict(X, verbose=0)[0]

        label = "VIOLENCE" if np.argmax(pred) == 1 else "NORMAL"
        confidence = np.max(pred) * 100

        if label == "VIOLENCE" and confidence > 80 and not alert_sent:
            send_telegram_alert(f"🚨 Violence detected ({confidence:.1f}%)")
            alert_sent = True

        frames.pop(0)

        cv2.putText(frame, f"{label} {confidence:.1f}%",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255) if label=="VIOLENCE" else (0,255,0), 2)

    cv2.imshow("CCTV Violence Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
