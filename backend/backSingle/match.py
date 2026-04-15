from flask import Flask, request, render_template
import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import gdown
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no display needed
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model from Google Drive if not present
model_drive_url = "https://drive.google.com/uc?id=1GjDUWzrnpVsiCW9xEXg8AV3CxRgdGxxN"
if not os.path.exists(MODEL_PATH):
    print("Downloading model.h5 from Google Drive...")
    gdown.download(model_drive_url, MODEL_PATH, quiet=False, fuzzy=True)

class DistanceLayer(Layer):
    def call(self, anchor, compare):
        sum_squared = tf.reduce_sum(tf.square(anchor - compare), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

# Load model using os.path — cross-platform, no hardcoded backslashes
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"DistanceLayer": DistanceLayer})

def preprocess_image(image_path, img_size=(64, 64)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.resize(image, img_size)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32) / 255.0

def compare_images(img1, img2):
    return model.predict([img1, img2])[0][0]

def visualize_result(image1_path, image2_path, similarity_score, threshold=0.3,
                     output_path="static/output_comparison.png"):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    if image1 is None or image2 is None:
        raise ValueError("Could not read one or both images.")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    label = "Same Person" if similarity_score > threshold else "Different Person"
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis("off")
    plt.title("IMAGE 1")
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis("off")
    plt.title("IMAGE 2")
    plt.suptitle(f"Result: {label} (Score: {similarity_score:.2f})", fontsize=16)
    plt.savefig(output_path)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare_faces():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return "Both images must be uploaded.", 400
        image1 = request.files['image1']
        image2 = request.files['image2']
        img1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
        img2_path = os.path.join(UPLOAD_FOLDER, image2.filename)
        image1.save(img1_path)
        image2.save(img2_path)
        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)
        score = compare_images(img1, img2)
        output_path = os.path.join('static', 'output_comparison.png')
        visualize_result(img1_path, img2_path, score, output_path=output_path)
        result_label = ("MATCHED! THE IMAGES ARE OF THE SAME PERSON"
                        if score >= 0.3 else
                        "NOT MATCHED! THE IMAGES ARE NOT OF THE SAME PERSON")
        return render_template('result.html',
                               image1_url=f"/{img1_path}",
                               image2_url=f"/{img2_path}",
                               similarity_score=f"{score:.2f}",
                               result_label=result_label,
                               output_image_url=f"/{output_path}")
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5003)
