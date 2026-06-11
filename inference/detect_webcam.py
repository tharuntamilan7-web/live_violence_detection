import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/violence_model.h5")
cap = cv2.VideoCapture(0)

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (128,128))
    frames.append(resized / 255.0)

    if len(frames) == 50:
        X = np.expand_dims(frames, axis=0)
        pred = model.predict(X, verbose=0)[0]
        label = "VIOLENCE" if np.argmax(pred)==1 else "NORMAL"
        frames.pop(0)

        cv2.putText(frame, label, (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,
                    (0,0,255) if label=="VIOLENCE" else (0,255,0),2)

    cv2.imshow("Webcam Violence Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
