
from flask import Flask, request, render_template
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import os
import torch
import gdown

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_DIR = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive model URLs (replace FILE_ID with actual Google Drive file IDs)
# facenet_drive_url = "https://drive.google.com/file/d/1PazvxYJBUtroAzDsTkPVv63GUnSI1VKX/view?usp=drive_link"
facenet_drive_url="https://drive.google.com/uc?export=download&id=1PazvxYJBUtroAzDsTkPVv63GUnSI1VKX"
# model_drive_url = "https://drive.google.com/uc?id=MODEL_H5_FILE_ID"

facenet_path = os.path.join(MODEL_DIR, "facenet_vggface2.pth")
# model_h5_path = os.path.join(MODEL_DIR, "model.h5")

# Download model if it does not exist
if not os.path.exists(facenet_path):
    print("Downloading FaceNet model...")
    gdown.download(facenet_drive_url, facenet_path, quiet=False)

# if not os.path.exists(model_h5_path):
#     print("Downloading model.h5...")
#     gdown.download(model_drive_url, model_h5_path, quiet=False)

# Load the FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def preprocess_image(image_file, img_size=(160, 160)):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1 / torch.norm(embedding1)
    embedding2 = embedding2 / torch.norm(embedding2)
    return torch.dot(embedding1.squeeze(), embedding2.squeeze()).item()

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

        embedding1 = facenet(img1)
        embedding2 = facenet(img2)

        similarity_score = cosine_similarity(embedding1, embedding2)
        threshold = 0.4
        result_label = "MATCHED! SAME PERSON" if similarity_score >= threshold else "NOT MATCHED! DIFFERENT PEOPLE"

        return render_template(
            'result.html',
            image1_url=f"/{img1_path}",
            image2_url=f"/{img2_path}",
            similarity_score=f"{similarity_score:.2f}",
            result_label=result_label
        )

    except Exception as e:
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)

# from flask import Flask, request, render_template
# from facenet_pytorch import InceptionResnetV1
# from torchvision import transforms
# from PIL import Image
# import os
# import torch

# app = Flask(__name__)

# # Directory to save uploaded images
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# facenet = InceptionResnetV1(pretrained='vggface2').eval()

# def preprocess_image(image_file, img_size=(160, 160)):
#     image = Image.open(image_file).convert("RGB")
#     transform = transforms.Compose([
#         transforms.Resize(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     return transform(image).unsqueeze(0)

# def cosine_similarity(embedding1, embedding2):
#     embedding1 = embedding1 / torch.norm(embedding1)
#     embedding2 = embedding2 / torch.norm(embedding2)
#     return torch.dot(embedding1.squeeze(), embedding2.squeeze()).item()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/compare', methods=['POST'])
# def compare_faces():
#     try:
#         if 'image1' not in request.files or 'image2' not in request.files:
#             return "Both images must be uploaded.", 400

#         # Save uploaded images
#         image1 = request.files['image1']
#         image2 = request.files['image2']

#         img1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
#         img2_path = os.path.join(UPLOAD_FOLDER, image2.filename)

#         image1.save(img1_path)
#         image2.save(img2_path)

#         # Preprocess images
#         img1 = preprocess_image(img1_path)
#         img2 = preprocess_image(img2_path)

#         # Generate embeddings
#         embedding1 = facenet(img1)
#         embedding2 = facenet(img2)

#         # Calculate similarity
#         similarity_score = cosine_similarity(embedding1, embedding2)
#         threshold = 0.4
#         result_label = "MATCHED! THE IMAGES ARE OF THE SAME PERSON" if similarity_score >= threshold else "NOT MATCHED! THE IMAGES ARE NOT OF THE SAME PERSON"

#         # Render results with images
#         return render_template(
#             'result.html',
#             image1_url=f"/{img1_path}",
#             image2_url=f"/{img2_path}",
#             similarity_score=f"{similarity_score:.2f}",
#             result_label=result_label
#         )

#     except Exception as e:
#         return f"Error: {e}", 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5002)


