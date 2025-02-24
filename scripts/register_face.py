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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # Match 5-inch TFT resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Tkinter GUI setup
root = tk.Tk()
root.title("Face Registration")
root.geometry("800x480")  # Adjusted for 5-inch TFT (800x480)
root.attributes('-fullscreen', True)  # Fullscreen for Raspberry Pi TFT

# Frame to hold the webcam feed
video_frame = tk.Frame(root, bg="black")
video_frame.pack(side=tk.TOP, pady=10, fill=tk.BOTH, expand=True)

# Label to display the webcam feed
video_label = tk.Label(video_frame, bg="black")
video_label.pack(fill=tk.BOTH, expand=True)

# Frame to hold the input fields and button
input_frame = tk.Frame(root, bg="white", padx=20, pady=10)
input_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Configure grid to center widgets
input_frame.columnconfigure(0, weight=1)
input_frame.columnconfigure(1, weight=1)
input_frame.rowconfigure(0, weight=1)
input_frame.rowconfigure(1, weight=1)
input_frame.rowconfigure(2, weight=1)

# Entry fields for user ID and name
tk.Label(input_frame, text="User ID (Numeric):", font=("Arial", 12)).grid(row=0, column=0, columnspan=2, pady=5, sticky="nsew")
user_id_entry = tk.Entry(input_frame, font=("Arial", 12), justify="center", width=20)
user_id_entry.grid(row=1, column=0, columnspan=2, pady=5, sticky="nsew")

tk.Label(input_frame, text="User Name:", font=("Arial", 12)).grid(row=2, column=0, columnspan=2, pady=5, sticky="nsew")
user_name_entry = tk.Entry(input_frame, font=("Arial", 12), justify="center", width=20)
user_name_entry.grid(row=3, column=0, columnspan=2, pady=5, sticky="nsew")

# Counter for the number of images saved
count = 0

# Function to capture and save face images
def register_face():
    global count
    user_id = user_id_entry.get().strip()
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

        # Convert to grayscale for detection and saving
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            messagebox.showerror("Error", "No face detected! Please adjust your position.")
            register_button.config(state=tk.NORMAL)
            return

        for (x, y, w, h) in faces:
            # Save the captured face as grayscale
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f'dataset/{user_name}_{user_id}_{count}.jpg', face_img)

            # Draw rectangle on the original frame for display
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display instructions
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

        cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert only for Tkinter display (BGR to RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        root.update()  # Update GUI
        cv2.waitKey(2000)  # Wait 2 seconds between captures

    # Enable the register button after capturing
    register_button.config(state=tk.NORMAL)
    messagebox.showinfo("Success", f"10 face images saved for {user_name} (ID: {user_id}).")

# Function to update the webcam feed in the Tkinter GUI
def update_frame():
    ret, frame = cap.read()
    if ret:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangle and instructions on the original frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Position your face here", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert to RGB only for Tkinter display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Repeat every 10 milliseconds
    video_label.after(10, update_frame)

# Register button
register_button = tk.Button(input_frame, text="Register Face", command=register_face, bg="green", fg="white", font=("Arial", 12))
register_button.grid(row=4, column=0, columnspan=2, pady=10, sticky="nsew")

# Start updating the webcam feed
update_frame()

# Run the Tkinter main loop
root.mainloop()

# Release the webcam when the app is closed
cap.release()
cv2.destroyAllWindows()