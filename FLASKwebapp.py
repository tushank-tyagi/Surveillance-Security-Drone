from flask import Flask, Response, render_template, request, send_from_directory
import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
import threading

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
RECORDING_FOLDER = 'recording'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RECORDING_FOLDER, exist_ok=True)

# Load the custom YOLOv10 model
model = YOLO(r'Model\test1.pt')

# Initialize variables for recording
recording = False
video_writer = None
recording_lock = threading.Lock()

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, conf=0.5)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    ret, jpeg = cv2.imencode('.jpg', annotated_frame)
    return jpeg.tobytes()

def gen_frames():
    global recording, video_writer
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_to_send = process_frame(frame)
            if recording:
                with recording_lock:
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
                        video_writer = cv2.VideoWriter(os.path.join(RECORDING_FOLDER, 'recorded_video.mp4'), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    video_writer.write(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
    cap.release()
    if video_writer:
        video_writer.release()

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    with recording_lock:
        recording = True
    return '', 204

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, video_writer
    with recording_lock:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None
    return '', 204

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return 'No file part', 400
    file = request.files['video_file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        process_uploaded_video(filepath)
        return '', 204

def process_uploaded_video(filepath):
    output_path = os.path.join(PROCESSED_FOLDER, 'processed_video.avi')
    cap = cv2.VideoCapture(filepath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, conf=0.5)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame)
    cap.release()
    out.release()

@app.route('/download_processed_video')
def download_processed_video():
    return send_from_directory(PROCESSED_FOLDER, 'processed_video.avi')

if __name__ == '__main__':
    app.run(debug=True)
