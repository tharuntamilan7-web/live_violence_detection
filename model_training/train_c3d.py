import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Dense, Dropout, GlobalAveragePooling3D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ================= GPU SETUP (SAFE FOR WINDOWS) =================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU enabled:", gpus)
    except:
        print("⚠️ GPU found but memory growth not enabled")
else:
    print("⚠️ No GPU found, running on CPU")

# ================= CONFIG =================
FRAME_DIR = "frames/train"
SEQ_LEN = 50
IMG_SIZE = 128
NUM_CLASSES = 2
BATCH_SIZE = 1          # MUST stay 1 on Windows
EPOCHS = 10

# ================= DATA GENERATOR =================
def video_generator():
    classes = ["non_violent", "violent"]

    while True:
        for label, cls in enumerate(classes):
            class_path = os.path.join(FRAME_DIR, cls)

            for video in os.listdir(class_path):
                video_path = os.path.join(class_path, video)

                if not os.path.isdir(video_path):
                    continue

                frames = sorted(os.listdir(video_path))
                if len(frames) < SEQ_LEN:
                    continue

                clip = []
                for f in frames[:SEQ_LEN]:
                    img_path = os.path.join(video_path, f)
                    img = cv2.imread(img_path)

                    if img is None:
                        continue

                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    clip.append(img / 255.0)

                if len(clip) != SEQ_LEN:
                    continue

                X = np.expand_dims(np.array(clip), axis=0)
                y = to_categorical([label], NUM_CLASSES)

                yield X, y

# ================= MODEL =================
model = Sequential([
    Conv3D(32, 3, activation='relu', padding='same',
           input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
    MaxPool3D((1, 2, 2)),

    Conv3D(64, 3, activation='relu', padding='same'),
    MaxPool3D((2, 2, 2)),

    Conv3D(128, 3, activation='relu', padding='same'),
    GlobalAveragePooling3D(),

    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ================= TRAIN =================
num_videos = sum(
    len(os.listdir(os.path.join(FRAME_DIR, c)))
    for c in ["non_violent", "violent"]
)

print(f"🎬 Training on {num_videos} videos")

model.fit(
    video_generator(),
    steps_per_epoch=num_videos,
    epochs=EPOCHS,
    verbose=1
)

# ================= SAVE =================
os.makedirs("models", exist_ok=True)
model.save("models/violence_model.h5")
print("✅ Model saved successfully")