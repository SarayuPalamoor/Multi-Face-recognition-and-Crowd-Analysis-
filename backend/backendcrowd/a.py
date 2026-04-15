from flask import Flask, request, render_template, url_for, send_from_directory
from ultralytics import YOLO
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Use yolov8n.pt — standard pretrained model, downloads automatically on first run
# This detects the 'person' class (class 0 in COCO) which is what we need for crowd analysis
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def crowd_analysis():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    results = model.predict(source=file_path, conf=0.4)

    # Count people (class 0 in COCO dataset)
    person_class_id = 0
    person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == person_class_id)

    return render_template("result.html",
                           image_url=url_for('uploaded_file', filename=file.filename),
                           person_count=person_count)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
