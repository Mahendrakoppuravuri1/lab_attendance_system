import cv2
import numpy as np
import face_recognition
import glob

# Helper function to load multiple images and calculate average encoding
def load_face_encodings(images_folder):
    encodings = []
    for img_path in glob.glob(f"{images_folder}/*.jpg"):  # Assuming jpg format, can be png as well
        image = face_recognition.load_image_file(img_path)
        try:
            encoding = face_recognition.face_encodings(image)[0]  # Get first face encoding
            encodings.append(encoding)
        except IndexError:
            print(f"Face not found in {img_path}")
            continue
    return np.mean(encodings, axis=0) if encodings else None

# Load known faces and encodings from multiple images
known_faces = {
    "Koppuravuri Venkata Naga Sai Mahendra": {
        "encoding": load_face_encodings("person1_images"),  # Folder with multiple images of person1
        "department": "NWC",
        "registration_number": "RA2211028010158"
    },
    "Maddu Rakesh": {
        "encoding": load_face_encodings("person2_images"),  # Folder with multiple images of person2
        "department": "NWC",
        "registration_number": "RA2211028010159"
    },
    "Konijeti Sai Kalyan": {
        "encoding": load_face_encodings("person3_images"),  # Folder with multiple images of person3
        "department": "NWC",
        "registration_number": "RA2211028010160"
    }
}

# Function to recognize faces
def recognize_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_locations, face_encodings

# Function to compare faces
def is_match(known_encoding, test_encoding, threshold=0.5):  # Adjusted threshold for stricter match
    return face_recognition.compare_faces([known_encoding], test_encoding, tolerance=threshold)[0]

# Recognize face function
# Recognize face function
def recognize_face(frame, face_locations):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    recognized_names = []

    if not face_encodings:  # Check if there are no face encodings
        return recognized_names

    for face_location, test_encoding in zip(face_locations, face_encodings):
        found = False
        for name, details in known_faces.items():
            known_encoding = details['encoding']

            if known_encoding is None:
                continue

            if is_match(known_encoding, test_encoding):
                recognized_names.append((name, details['department'], details['registration_number'], face_location))
                found = True
                print(f"Recognized {name}")
                break
        
        if not found:
            recognized_names.append(("Unknown", None, None, face_location))
            print("Unknown face detected")

    return recognized_names

# Main function for testing
def main():
    # Initialize the video capture from the camera
    video_capture = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = video_capture.read()

            if not ret:
                print("Failed to capture image.")
                break

            # Recognize faces from the captured frame
            recognized_faces = recognize_face(frame)

            # Draw boxes around recognized faces
            for name, department, reg_number, location in recognized_faces:
                top, right, bottom, left = location
                color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)  # Green for recognized, Red for unknown
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Release the capture and close windows
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
