from flask import Flask, render_template, request
import subprocess
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

process = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_video", methods=["POST"])
def start_video():
    global process
    video = request.files["video"]
    path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(path)

    process = subprocess.Popen(["python", "inference/detect_video.py", path])
    return "✅ Video detection started"

@app.route("/start_webcam")
def start_webcam():
    global process
    process = subprocess.Popen(["python", "inference/detect_webcam.py"])
    return "✅ Webcam started"

@app.route("/start_cctv", methods=["POST"])
def start_cctv():
    global process
    rtsp = request.form["rtsp"]
    process = subprocess.Popen(["python", "inference/detect_cctv.py", rtsp])
    return "✅ CCTV started"

@app.route("/stop")
def stop():
    global process
    if process:
        process.terminate()
        process = None
    return "🛑 Stopped"

if __name__ == "__main__":
    app.run(debug=True)
