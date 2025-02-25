import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# Create dataset folder if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Tkinter GUI setup
root = tk.Tk()
root.title("Face Registration")
root.geometry("800x800")

# Frame to hold the webcam feed
video_frame = tk.Frame(root, bg="black")  # Set background to black for better contrast
video_frame.pack(side=tk.TOP, pady=10, fill=tk.BOTH, expand=True)

# Label to display the webcam feed
video_label = tk.Label(video_frame, bg="black")
video_label.pack(fill=tk.BOTH, expand=True)

# Frame to hold the input fields and button
input_frame = tk.Frame(root, bg="white", padx=20, pady=20)  # Add padding for spacing
input_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Configure grid to center widgets
input_frame.columnconfigure(0, weight=1)  # Center column
input_frame.columnconfigure(1, weight=1)  # Center column
input_frame.rowconfigure(0, weight=1)     # Center row
input_frame.rowconfigure(1, weight=1)     # Center row
input_frame.rowconfigure(2, weight=1)     # Center row

# Entry fields for user ID and name
tk.Label(input_frame, text="User ID (Numeric):", font=("Arial", 12)).grid(row=0, column=0, columnspan=2, pady=5, sticky="nsew")
user_id_entry = tk.Entry(input_frame, font=("Arial", 12), justify="center", width=20)  # Shorter width
user_id_entry.grid(row=1, column=0, columnspan=2, pady=5, sticky="nsew")

tk.Label(input_frame, text="User Name:", font=("Arial", 12)).grid(row=2, column=0, columnspan=2, pady=5, sticky="nsew")
user_name_entry = tk.Entry(input_frame, font=("Arial", 12), justify="center", width=20)  # Shorter width
user_name_entry.grid(row=3, column=0, columnspan=2, pady=5, sticky="nsew")

# Counter for the number of images saved
count = 0

# Function to capture and save face images
def register_face():
    global count
    user_id = user_id_entry.get().strip()  # Remove any extra spaces
    user_name = user_name_entry.get().strip()

    # Validate inputs
    if not user_id or not user_name:
        messagebox.showerror("Error", "Please enter both User ID and Name!")
        return

    if not user_id.isdigit():
        messagebox.showerror("Error", "User ID must be numeric!")
        return

    # Disable the register button during capture
    register_button.config(state=tk.DISABLED)

    # Capture 10 images with instructions
    for i in range(1, 11):
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image!")
            register_button.config(state=tk.NORMAL)
            return

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            messagebox.showerror("Error", "No face detected! Please adjust your position.")
            register_button.config(state=tk.NORMAL)
            return

        for (x, y, w, h) in faces:
            # Save the captured face into the dataset folder
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f'dataset/{user_name}_{user_id}_{count}.jpg', face_img)

        # Display instructions for the user
        if i == 1:
            instruction = "Look straight"
        elif i == 2:
            instruction = "Tilt your head slightly left"
        elif i == 3:
            instruction = "Tilt your head slightly right"
        elif i == 4:
            instruction = "Tilt your head slightly up"
        elif i == 5:
            instruction = "Tilt your head slightly down"
        else:
            instruction = "Look straight"

        # Show the instruction on the screen
        cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Registration", frame)
        cv2.waitKey(2000)  # Wait for 1 second to allow the user to adjust

    # Enable the register button after capturing
    register_button.config(state=tk.NORMAL)
    messagebox.showinfo("Success", f"10 face images saved for {user_name} (ID: {user_id}).")

# Function to update the webcam feed in the Tkinter GUI
def update_frame():
    ret, frame = cap.read()
    if ret:
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangle around the face and display instructions
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Position your face here", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame to a format Tkinter can display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Repeat every 10 milliseconds
    video_label.after(10, update_frame)

# Register button
register_button = tk.Button(input_frame, text="Register Face", command=register_face, bg="green", fg="white", font=("Arial", 12))
register_button.grid(row=4, column=0, columnspan=2, pady=20, sticky="nsew")

# Start updating the webcam feed
update_frame()

# Run the Tkinter main loop
root.mainloop()

# Release the webcam when the app is closed
cap.release()
cv2.destroyAllWindows()