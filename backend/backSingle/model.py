import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Load Dataset
faces = 'olivetti_faces.npy'
faces_targets = 'olivetti_faces_target.npy'

# Function to generate image pairs
def generate_image_pairs(images, labels):
    # Generate index for each label
    unique_labels = np.unique(labels)
    label_wise_indices = {label: [index for index, curr_label in enumerate(labels) if label == curr_label] for label in unique_labels}
    
    # Generate image pairs and labels
    pair_images = []
    pair_labels = []
    for index, image in enumerate(images):
        pos_indices = label_wise_indices.get(labels[index])
        pos_image = images[np.random.choice(pos_indices)]
        pair_images.append((image, pos_image))
        pair_labels.append(1)

        neg_indices = np.where(labels != labels[index])
        neg_image = images[np.random.choice(neg_indices[0])]
        pair_images.append((image, neg_image))
        pair_labels.append(0)
        
    return np.array(pair_images), np.array(pair_labels)

# Load the images and labels
face_images = np.load(faces)
face_labels = np.load(faces_targets)
target_shape = face_images[0].shape
images_dataset, labels_dataset = generate_image_pairs(face_images, face_labels)
images_dataset, labels_dataset = shuffle(images_dataset, labels_dataset)

# Siamese Network Embedding Model
inputs = layers.Input((64, 64, 1))

x = layers.Conv2D(64, (10, 10), padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(128, (7, 7), padding="same", activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(128, (4, 4), padding="same", activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(256, (4, 4), padding="same", activation="relu")(x)
fcOutput = layers.Flatten()(x)
fcOutput = layers.Dense(4096, activation="relu")(fcOutput)
outputs = layers.Dense(1024, activation="sigmoid")(fcOutput)

embedding = Model(inputs, outputs, name="Embedding")

# Contrastive Loss Function
import tensorflow.keras.backend as K
def contrastive_loss(y, preds, margin=1):
    y = tf.cast(y, preds.dtype)
    squared_preds = K.square(preds)
    squared_margin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squared_preds + (1 - y) * squared_margin)
    return loss

# Distance Layer
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance
    between the embeddings
    """
    def _init_(self, **kwargs):
        super()._init_(**kwargs)

    def call(self, anchor, compare):
        sum_squared = K.sum(K.square(anchor - compare), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))

# Define the Inputs for the Siamese Network
anchor_input = layers.Input(name="anchor", shape=target_shape + (1,))
compare_input = layers.Input(name="compare", shape=target_shape + (1,))

# Get the distances between the two inputs (anchor and compare)
distances = DistanceLayer()(embedding(anchor_input), embedding(compare_input))

# Output layer for classification (similar or not)
outputs = layers.Dense(1, activation="sigmoid")(distances)

# Build the Siamese model
siamese_model = Model(inputs=[anchor_input, compare_input], outputs=outputs)

# Compile the model
siamese_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = siamese_model.fit([images_dataset[:, 0, :], images_dataset[:, 1, :]], labels_dataset,
                            epochs=150, validation_split=0.2, batch_size=64)

# Save the trained model
siamese_model.save('model.h5')

# Output the training history for further analysis
print("Training completed. Model saved as 'model.h5'.")