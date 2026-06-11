🚨 Live Violence Detection System
📖 Overview

A Deep Learning-based surveillance system that detects violent activities in real-time using 3D CNN, TensorFlow, and OpenCV. The system supports webcam, CCTV (RTSP streams), and uploaded videos, and sends instant Telegram alerts when violence is detected.

✨ Features
🎥 Real-time Webcam Monitoring
📹 CCTV (RTSP) Stream Support
📂 Video Upload Detection
🧠 3D CNN-based Violence Classification
🚨 Telegram Alert Notifications
🌐 Flask Web Dashboard
⚡ Real-Time Prediction with Confidence Score

🛠 Tech Stack
Python
TensorFlow / Keras
OpenCV
Flask
NumPy
Telegram Bot API
HTML / CSS

📂 Project Structure
live_violence_detection/
│
├── alerts/
├── data/raw_videos/
├── frames/train/
├── inference/
├── model_training/
├── models/
├── static/
├── templates/
├── uploads/
├── utils/
│
├── app.py
└── requirements.txt

🔄 Workflow
Video Dataset
      │
      ▼
Frame Extraction
      │
      ▼
Preprocessing
      │
      ▼
3D CNN Training
      │
      ▼
Model Saved (.h5)
      │
      ▼
Real-Time Detection
(Webcam/CCTV/Video)
      │
      ▼
Telegram Alert
🧠 Model Architecture
Input Video Frames
        │
        ▼
Conv3D
        │
        ▼
MaxPool3D
        │
        ▼
Conv3D
        │
        ▼
GlobalAveragePooling3D
        │
        ▼
Dense Layer
        │
        ▼
Softmax Output
(Violent / Non-Violent)

🚀 Installation
Clone Repository
git clone https://github.com/yourusername/live-violence-detection.git
cd live-violence-detection
Install Dependencies
pip install -r requirements.txt

📊 Training
Extract Frames
python utils/extract_frames.py
Train Model
python model_training/train_c3d.py
Saved Model
models/violence_model.h5

🎥 Run Detection
Webcam
python inference/detect_webcam.py
Video File
python inference/detect_video.py
CCTV Stream
python inference/detect_cctv.py

🌐 Run Web Application
python app.py

Open:

http://127.0.0.1:5000
🚨 Telegram Alert

Example Alert:

🚨 Violence Detected!

Camera: CCTV-01
Confidence: 92.5%
Time: 14:30:15

📈 Future Enhancements
Email Alerts
Weapon Detection
Face Recognition
Multi-Camera Monitoring
Cloud Deployment
Mobile Application

👨‍💻 Author
Tharun
B.E. Electrical and Electronics Engineering (EEE)

⭐ Project Highlights
Deep Learning-Based Surveillance
Real-Time Violence Detection
CCTV & Webcam Integration
Telegram Alert System
Flask Web Dashboard
