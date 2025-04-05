# app.py
import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Maximum allowed file size: 50MB
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Define upload folder inside static
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions for images and videos
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}

# Load the trained deepfake detection model
MODEL_PATH = "deepfake_model.h5"
model = load_model(MODEL_PATH)

# Target image size (must match training)
IMG_HEIGHT, IMG_WIDTH = 224, 224

def allowed_file(filename, input_type):
    """Check if the file extension is allowed for the selected input type."""
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    if input_type == "image":
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif input_type == "video":
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False

def preprocess_img_cv2(filepath):
    """Load and preprocess an image using OpenCV."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Unable to load image from {filepath}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_img_pil(url):
    """Download, open, and preprocess an image from a URL using PIL."""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(img_array):
    """Run the model on the preprocessed image array and return the prediction."""
    prediction = model.predict(img_array)[0][0]
    return prediction

def process_video(video_path, frame_interval=30):
    """
    Process a video file by extracting every 'frame_interval'-th frame,
    predicting each frame, and averaging predictions.
    """
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = np.expand_dims(img, axis=0) / 255.0
                pred = predict_image(img)
                frame_predictions.append(pred)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
        frame_count += 1
    cap.release()
    if frame_predictions:
        return np.mean(frame_predictions)
    else:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    file_url = None

    if request.method == "POST":
        # Get input type: "image" or "video"
        input_type = request.form.get("input_type")
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return "No file selected."
            if file and allowed_file(file.filename, input_type):
                filename = file.filename
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                if input_type == "image":
                    try:
                        img_array = preprocess_img_cv2(filepath)
                        prediction = predict_image(img_array)
                        result = "Real Image" if prediction > 0.5 else "Deepfake Detected"
                        confidence = prediction if prediction > 0.5 else 1 - prediction
                        file_url = filepath
                    except Exception as e:
                        return f"Error processing image: {str(e)}"
                elif input_type == "video":
                    video_prediction = process_video(filepath, frame_interval=30)
                    if video_prediction is not None:
                        result = "Real video" if video_prediction > 0.5 else "Deepfake Detected"
                        confidence = video_prediction if video_prediction > 0.5 else 1 - video_prediction
                        file_url = filepath  # For videos, consider extracting a thumbnail later.
                    else:
                        return "Error processing video file."
            else:
                return "Invalid file type for selected input."
        else:
            return "No file uploaded."

    return render_template("index.html", result=result,
                           confidence=(round(confidence * 100, 2) if confidence is not None else None),
                           file_url=file_url)

if __name__ == "__main__":
    app.run(debug=True)
