import cv2
import face_recognition
import numpy as np
import os
import time

# Load registered face encodings and names
def load_registered_faces(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            image = face_recognition.load_image_file(os.path.join(dataset_path, filename))
            # Encode the face
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) > 0:  # Ensure a face was found in the image
                face_encoding = face_encoding[0]
                # Extract the name from the filename (format: name_id_count.jpg)
                name = filename.split('_')[0]  # Extract the first part (name)
                # Add to the known faces list
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

    return known_face_encodings, known_face_names

# Load registered faces
dataset_path = "dataset"
known_face_encodings, known_face_names = load_registered_faces(dataset_path)

# Initialize webcam
cap = cv2.VideoCapture(0)

# FPS variables
fps_start_time = time.time()
fps_counter = 0
fps = 0

# Frame skipping variables
frame_skip = 2  # Process every 2nd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Increment frame counter
    frame_count += 1

    # Skip frames to improve FPS
    if frame_count % frame_skip != 0:
        continue

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce resolution
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame using HOG model (faster than CNN)
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        accuracy = "0%"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # Calculate accuracy (confidence score) as a percentage
            accuracy = f"{round((1 - face_distances[best_match_index]) * 100, 2)}%"

        # Scale the face locations back to the original frame size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the name and accuracy below the face
        label = f"{name} ({accuracy})"
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1:  # Update FPS every second
        fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()

    # Display FPS on the top-right corner
    cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Attendee Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()