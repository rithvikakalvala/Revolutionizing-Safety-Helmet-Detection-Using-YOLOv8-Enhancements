from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Folder for uploaded images and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load the YOLO model
model = YOLO('best.pt')  # Path to your trained YOLO weights

@app.route("/")
def index():
    return render_template("home.html")



@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('abstract.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video formats
            result_video_path = process_video(file_path, filename)
            return render_template('result.html', 
                                   original_file=filename, 
                                   result_video=result_video_path)

        else:  # Image file
            results = model(file_path)
            result_img_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
            annotated_frame = results[0].plot()
            cv2.imwrite(result_img_path, annotated_frame)
            return render_template('result.html', 
                                   original_file=filename, 
                                   result_file=f"result_{filename}")
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/results/videos/<filename>')
def result_video(filename):
    video_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, mimetype='video/mp4')


def process_video(input_path, filename):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec for better compatibility
    output_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO processing
        results = model(frame)
        annotated_frame = results[0].plot()  # Annotate frame
        out.write(annotated_frame)

    cap.release()
    out.release()
    return f"result_{filename}"  # Return relative path


if __name__ == '__main__':
    app.run(debug=True)
