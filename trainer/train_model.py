import cv2
import os
import numpy as np
from PIL import Image

dataset_path = 'dataset'

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to get images and labels
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    labels = []

    for image_path in image_paths:
        # Convert image to grayscale
        face_img = Image.open(image_path).convert('L')
        face_np = np.array(face_img, 'uint8')

        # Extract user ID from the image name
        user_id = int(os.path.split(image_path)[-1].split('_')[1])
        faces.append(face_np)
        labels.append(user_id)

    return faces, labels

# Get faces and labels
faces, labels = get_images_and_labels(dataset_path)

# Train the model
recognizer.train(faces, np.array(labels))

# Save the trained model
if not os.path.exists('trainer'):
    os.makedirs('trainer')
recognizer.save('trainer/trainer.yml')
print("Training completed. Model saved to 'trainer/trainer.yml'.")