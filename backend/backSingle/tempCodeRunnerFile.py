import os
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define the DistanceLayer for loading
class DistanceLayer(Layer):
    def call(self, anchor, compare):
        sum_squared = tf.reduce_sum(tf.square(anchor - compare), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

# Load the model with custom layer
model = load_model('model.h5', custom_objects={"DistanceLayer": DistanceLayer})

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to serve the HTML page for uploading images
@app.route('/')
def index():
    return render_template('index.html')

# Preprocess the image for prediction
def preprocess_image(file, img_size=(64, 64)):
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid image file")
    image = cv2.resize(image, img_size)  # Resize the image to match the model's input size
    image = np.expand_dims(image, axis=-1)  # Add the channel dimension (for grayscale images)
    image = np.expand_dims(image, axis=0)  # Add the batch dimension
    return image.astype(np.float32) / 255.0  # Normalize the image

# Visualize the comparison result
def visualize_result(image1_file, image2_file, similarity_score, threshold=0.3, output_path="static/output_comparison.png"):
    try:
        # Convert file-like object to OpenCV image
        image1 = cv2.imdecode(np.frombuffer(image1_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(image2_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image1 is None or image2 is None:
            raise ValueError("Failed to load one or both images for visualization.")

        # Convert BGR to RGB for display
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        # Determine the result based on the threshold
        result_label = "Same Person" if similarity_score > threshold else "Different Person"

        # Plot the images side by side
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image1)
        plt.axis("off")
        plt.title("IMAGE 1")

        plt.subplot(1, 2, 2)
        plt.imshow(image2)
        plt.axis("off")
        plt.title("IMAGE 2")

        # Add the result label as the overall title
        plt.suptitle(f"Result: {result_label} (Score: {similarity_score:.2f})", fontsize=16, color="black")

        # Save the visualization to a file
        plt.savefig(output_path)
        print(f"Comparison visualization saved to {output_path}")
        plt.close()

    except Exception as e:
        print(f"Error during visualization: {e}")


# Define the route to handle the image upload and comparison
@app.route('/upload', methods=['POST'])
def upload_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    image1_file = request.files['image1']
    image2_file = request.files['image2']

    try:
        # Generate unique filenames based on current time and original extension
        image1_filename = os.path.join(UPLOAD_FOLDER, f"image1_{int(time.time())}.jpg")
        image2_filename = os.path.join(UPLOAD_FOLDER, f"image2_{int(time.time())}.jpg")

        # Save the uploaded files with unique names
        image1_file.save(image1_filename)
        image2_file.save(image2_filename)

        # Preprocess the images
        img1 = preprocess_image(image1_file)
        img2 = preprocess_image(image2_file)

        # Predict the similarity score between the two images
        score = model.predict([img1, img2])[0][0]

        # Print the result based on the threshold
        result = "MATCHED! THE IMAGES ARE OF THE SAME PERSON" if score >= 0.3 else "NOT MATCHED! THE IMAGES ARE NOT OF THE SAME PERSON"
        
        # Visualize the comparison
        output_path = "static/output_comparison.png"
        visualize_result(image1_file, image2_file, score, output_path=output_path)

        # Redirect to result page with the result message and image paths
        return redirect(url_for('result', result_message=result, score=score, image1_url=image1_filename, image2_url=image2_filename, image_path=output_path))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to display the result after uploading images
@app.route('/result')
def result():
    result_message = request.args.get('result_message')
    score = request.args.get('score')
    image1_url = request.args.get('image1_url')
    image2_url = request.args.get('image2_url')
    image_path = request.args.get('image_path')
    return render_template('result.html', result_message=result_message, score=score, image1_url=image1_url, image2_url=image2_url, image_path=image_path)


if __name__ == '__main__':
    app.run(debug=True, port=5003)