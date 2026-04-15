import sys
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
import shutil

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

detector = MTCNN()
embedder = FaceNet()

# encodings file saved in the same server/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encodings_file = os.path.join(BASE_DIR, "multiface1.pkl")

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def load_existing_encodings(encodings_file):
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    return [], []

def save_encodings(encodings_file, data):
    temp_file = encodings_file + ".tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(data, f)
    shutil.move(temp_file, encodings_file)

def generate_encodings(dataset_path, encodings_file):
    existing_encodings, existing_names = load_existing_encodings(encodings_file)
    new_encodings = []
    new_names = []

    if not os.path.isdir(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return

    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}. Skipping...")
            continue
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_image)
        if len(faces) == 0:
            print(f"No faces found in {image_path}. Skipping...")
            continue
        face = faces[0]
        x, y, width, height = face["box"]
        x, y = max(0, x), max(0, y)
        face_roi = rgb_image[y:y+height, x:x+width]
        preprocessed_face = preprocess_face(face_roi)
        embedding = embedder.model.predict(preprocessed_face)[0]
        new_encodings.append(embedding)
        new_names.append(os.path.basename(dataset_path))

    combined_encodings = existing_encodings + new_encodings
    combined_names = existing_names + new_names
    save_encodings(encodings_file, {"encodings": combined_encodings, "names": combined_names})
    print(f"Encodings updated and saved to {encodings_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_encodings.py <dataset_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    generate_encodings(dataset_path, encodings_file)
