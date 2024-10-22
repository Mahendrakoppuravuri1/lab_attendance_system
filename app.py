import cv2
import numpy as np
import face_recognition
import glob
from flask import Flask, render_template, Response, jsonify
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Global variable to control recognition state and tracking last recognized user
last_recognized = None

def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            registration_number TEXT NOT NULL,
            time_in TEXT,
            time_out TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()  # Initialize the database when the app starts

# Helper function to load multiple images and calculate average encoding
def load_face_encodings(images_folder):
    encodings = []
    for img_path in glob.glob(f"{images_folder}/*.jpg"):
        image = face_recognition.load_image_file(img_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            encodings.append(face_encodings[0])
        else:
            print(f"Face not found in {img_path}")
    return np.mean(encodings, axis=0) if encodings else None

# Load known faces and encodings from multiple images
known_faces = {
    "Koppuravuri Venkata Naga Sai Mahendra": {
        "encoding": load_face_encodings("person1_images"),
        "department": "NWC",
        "registration_number": "RA2211028010158"
    },
    "Maddu Rakesh": {
        "encoding": load_face_encodings("person2_images"),
        "department": "NWC",
        "registration_number": "RA2211028010159"
    },
    "Konijeti Sai Kalyan": {
        "encoding": load_face_encodings("person3_images"),
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
def is_match(known_encoding, test_encoding, threshold=0.5):
    return face_recognition.compare_faces([known_encoding], test_encoding, tolerance=threshold)[0]

# Function to log attendance
def log_attendance(name, department, registration_number):
    # Connect to the database
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Check if the person is already logged in
    cursor.execute("SELECT * FROM attendance WHERE name = ? AND time_out IS NULL", (name,))
    record = cursor.fetchone()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if record:
        # If already logged in, log them out
        cursor.execute("UPDATE attendance SET time_out = ? WHERE name = ? AND time_out IS NULL", 
                       (current_time, name))
        status = f"{name} logged out at {current_time}"
    else:
        # If not logged in, log them in
        cursor.execute("INSERT INTO attendance (name, department, registration_number, time_in) VALUES (?, ?, ?, ?)", 
                       (name, department, registration_number, current_time))
        status = f"{name} logged in at {current_time}"

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    return status

# Function to recognize a face
def recognize_face(frame, face_locations):
    global last_recognized
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    recognized_names = []

    if not face_encodings:
        return recognized_names

    for face_location, test_encoding in zip(face_locations, face_encodings):
        found = False
        for name, details in known_faces.items():
            known_encoding = details['encoding']
            if known_encoding is None:
                continue

            if is_match(known_encoding, test_encoding):
                recognized_names.append((name, details['department'], details['registration_number'], face_location))

                # If this is the same person recognized again
                if last_recognized and last_recognized[0] == name:
                    # Log out immediately if recognized again
                    log_attendance(name, details['department'], details['registration_number'])
                    print(f"{name} logged out.")  # Log the logout time
                    last_recognized = None  # Clear last recognized after logging out
                else:
                    # New recognition, log them in
                    log_attendance(name, details['department'], details['registration_number'])
                    print(f"{name} logged in.")  # Log the login time
                    last_recognized = (name, details['department'], details['registration_number'])
                
                found = True
                break
        
        if not found:
            recognized_names.append(("Unknown", None, None, face_location))
            print("Unknown face detected")

    return recognized_names

# Function to generate video frames
def gen_frames():
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        success, frame = video_capture.read()
        if not success:
            print("Error: Frame not captured")  # Debugging statement
            break

        recognized_faces = []

        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            recognized_faces = recognize_face(frame, face_locations)

        # Draw rectangles around recognized faces
        for name, department, registration, face_location in recognized_faces:
            top, right, bottom, left = face_location
            color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{name} ({department})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()  # Release the video capture on exit

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recognition', methods=['GET'])
def start_recognition():
    return jsonify({"message": "Recognition started"})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/last_recognized')
def last_recognized_info():
    global last_recognized  # Make sure this is defined elsewhere in your code

    if last_recognized:
        name = last_recognized[0]
        department = last_recognized[1]
        registration_number = last_recognized[2]
        
        # Log attendance and get the status
        status = log_attendance(name, department, registration_number)

        return jsonify({
            "name": name,
            "department": department,
            "registration": registration_number,
            "status": status  # Include the login/logout status
        })
    else:
        return jsonify({"status": "No face recognized yet"})

@app.route('/retry_recognition', methods=['GET'])
def retry_recognition():
    return jsonify({"message": "Recognition has been restarted."})

@app.route('/log_out', methods=['POST'])
def log_out():
    global last_recognized
    if last_recognized:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_attendance(last_recognized[0], last_recognized[1], last_recognized[2])  # Log time out
        message = f"{last_recognized[0]} logged out at {current_time}"
        last_recognized = None  # Clear last recognized after logging out
        return jsonify({"message": message})
    return jsonify({"message": "No user is currently recognized."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)