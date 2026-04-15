from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
from pymongo import MongoClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize MTCNN and FaceNet
detector = MTCNN()
embedder = FaceNet()

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["faces"]
collection = db["imagedocuments"]

# Paths — all relative now, no hardcoded Windows paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encodings_file = os.path.join(BASE_DIR, "multiface1.pkl")
output_image_path = os.path.join(BASE_DIR, "static", "output.jpg")
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)

# Load encodings if they exist
if not os.path.exists(encodings_file):
    print(f"WARNING: {encodings_file} not found. Run generate_encodings.py first.")
    known_encodings = np.array([])
    known_names = []
else:
    with open(encodings_file, "rb") as f:
        data = pickle.load(f)
        known_encodings = np.array(data["encodings"])
        known_names = data["names"]

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def retrieve_person_data(rollno):
    person_data = collection.find_one({"qrData.rollNo": rollno})
    if person_data:
        qr_data = person_data.get("qrData", {})
        return {
            "rollNo": qr_data.get("rollNo", "N/A"),
            "name": qr_data.get("name", "N/A"),
            "fatherName": qr_data.get("fatherName", "N/A"),
            "department": qr_data.get("department", "N/A"),
            "contact": qr_data.get("contact", "N/A"),
            "images_count": len(person_data.get("images", [])),
        }
    return {"rollNo": rollno, "name": "Unknown", "fatherName": "N/A",
            "department": "N/A", "contact": "N/A", "images_count": 0}

def recognize_faces(image_path, threshold=0.7):
    if len(known_encodings) == 0:
        return [{"rollNo": "No model", "name": "Run generate_encodings.py first"}]

    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)
    recognized_details = []

    for face in faces:
        x, y, width, height = face["box"]
        x, y = max(0, x), max(0, y)
        face_roi = rgb_image[y:y+height, x:x+width]
        preprocessed_face = preprocess_face(face_roi)
        embedding = embedder.model.predict(preprocessed_face)[0]
        distances = np.linalg.norm(known_encodings - embedding, axis=1)
        min_distance = np.min(distances)

        if min_distance < threshold:
            rollno = known_names[np.argmin(distances)]
            person_data = retrieve_person_data(rollno)
            recognized_details.append(person_data)
            label = rollno
        else:
            recognized_details.append({"rollNo": "Unknown", "name": "Unknown",
                                       "fatherName": "N/A", "department": "N/A",
                                       "contact": "N/A", "images_count": 0})
            label = "Unknown"

        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imwrite(output_image_path, image)
    return recognized_details

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    file_path = os.path.join(BASE_DIR, "uploaded_image.jpg")
    file.save(file_path)
    recognized_details = recognize_faces(file_path)
    return render_template('result.html',
                           image_url=url_for('static', filename='output.jpg'),
                           recognized_details=recognized_details)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
