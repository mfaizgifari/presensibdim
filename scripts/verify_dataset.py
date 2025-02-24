import cv2
import face_recognition
import os
import shutil

def verify_and_clean_dataset(dataset_path="dataset", cleaned_path="cleaned_dataset"):
    # Create a cleaned dataset directory
    if not os.path.exists(cleaned_path):
        os.makedirs(cleaned_path)

    valid_images = 0
    invalid_images = 0

    print("Verifying dataset images...")
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(dataset_path, filename)
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image, model="hog")

            if len(face_locations) == 1:  # Exactly one face should be detected
                # Move valid image to cleaned dataset
                shutil.copy(filepath, os.path.join(cleaned_path, filename))
                valid_images += 1
                print(f"Valid: {filename}")
            else:
                invalid_images += 1
                print(f"Invalid: {filename} - {len(face_locations)} faces detected")

    print(f"\nVerification complete!")
    print(f"Valid images: {valid_images}")
    print(f"Invalid images: {invalid_images}")
    print(f"Cleaned dataset saved to: {cleaned_path}")

if __name__ == "__main__":
    verify_and_clean_dataset()