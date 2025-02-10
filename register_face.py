import cv2
import os

# Create dataset folder if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Ask for user ID and name
user_id = input("Enter user ID: ")
user_name = input("Enter user name: ")

# Counter for the number of images saved
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Save the captured face into the dataset folder
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f'dataset/{user_name}_{user_id}_{count}.jpg', face_img)

        # Display the count of images saved
        cv2.putText(frame, f"Images Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Register Face', frame)

    # Break the loop if 50 images are captured or 'q' is pressed
    if count >= 50 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
print(f"Face registration completed for {user_name} (ID: {user_id}).")