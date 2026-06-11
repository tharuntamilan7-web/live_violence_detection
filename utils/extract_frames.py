import cv2
import os

# ================= CONFIG =================
RAW_DIR = "data/raw_videos"
OUT_DIR = "frames/train"
IMG_SIZE = 128
MAX_FRAMES = 50

os.makedirs(OUT_DIR, exist_ok=True)

# ================= EXTRACTION =================
for cls in ["violent", "non_violent"]:
    input_dir = os.path.join(RAW_DIR, cls)
    output_dir = os.path.join(OUT_DIR, cls)
    os.makedirs(output_dir, exist_ok=True)

    for video in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video)
        cap = cv2.VideoCapture(video_path)

        video_name = os.path.splitext(video)[0]
        save_dir = os.path.join(output_dir, video_name)
        os.makedirs(save_dir, exist_ok=True)

        frame_count = 0
        print(f"\n🎥 Processing: {video} [{cls}]")

        while cap.isOpened() and frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(f"{save_dir}/{frame_count}.jpg", frame)

            # 🔴 SHOW LIVE FRAME
            cv2.putText(
                frame,
                f"{cls.upper()} | Frame {frame_count+1}/{MAX_FRAMES}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if cls == "violent" else (0, 255, 0),
                2
            )

            cv2.imshow("Live Frame Extraction", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("❌ Extraction stopped bypython3 model_training/train_model.py user")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            frame_count += 1

        cap.release()
        print(f"✅ Saved {frame_count} frames")

cv2.destroyAllWindows()
print("\n🎉 ALL FRAME EXTRACTION COMPLETED")
