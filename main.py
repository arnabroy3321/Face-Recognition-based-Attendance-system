import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Paths
USER_DIR = "registered_users"
TRAINING_FILE = "trainer.yml"
EXCEL_FILE = "attendance.xlsx"

# Ensure the directory exists
if not os.path.exists(USER_DIR):
    os.makedirs(USER_DIR)

# Load OpenCV Face Recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to capture and save user images
def register_user(name):
    cap = cv2.VideoCapture(0)
    count = 0
    user_folder = os.path.join(USER_DIR, name)
    os.makedirs(user_folder, exist_ok=True)

    print(f"Capturing images for {name}. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            file_path = os.path.join(user_folder, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Register", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {name}. Now training...")

    train_model()

# Function to train the model
def train_model():
    faces = []
    labels = []
    label_dict = {}

    for idx, person in enumerate(os.listdir(USER_DIR)):
        person_path = os.path.join(USER_DIR, person)
        label_dict[idx] = person

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(idx)

    if faces and labels:
        recognizer.train(faces, np.array(labels))
        recognizer.save(TRAINING_FILE)
        print("Model trained successfully!")

# Function to recognize user and mark attendance
def recognize_user():
    if not os.path.exists(TRAINING_FILE):
        print("No trained model found. Register users first.")
        return

    recognizer.read(TRAINING_FILE)
    cap = cv2.VideoCapture(0)

    print("Recognizing... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)

            if confidence < 50:
                name = os.listdir(USER_DIR)[label]
                mark_attendance(name)
                return
            else:
                name = "Unknown"

        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #     cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # cv2.imshow("Recognition", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

# Function to mark attendance
def mark_attendance(name):
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        df = pd.read_excel(EXCEL_FILE)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    df = pd.concat([df, pd.DataFrame([[name, date_str, time_str]], columns=df.columns)], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    print(f"Attendance marked for {name} at {time_str}")

# Main program
if __name__ == "__main__":
    while True:
        choice = input("Enter 'r' to register, 'a' to mark attendance, or 'q' to quit: ")
        if choice == 'r':
            name = input("Enter your name: ")
            register_user(name)
        elif choice == 'a':
            recognize_user()
        elif choice == 'q':
            break
