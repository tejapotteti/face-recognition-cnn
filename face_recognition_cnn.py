import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array

# Step 1: Directory and Utility Setup
DATASET_DIR = "faces_dataset"
MODEL_PATH = "face_recognition_model.h5"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def capture_faces():
    name = input("Enter the name of the person: ").strip()
    person_dir = os.path.join(DATASET_DIR, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("Press 'q' to stop capturing images manually.")
    print("Capturing images... Please look at the camera.")

    count = 0
    target_images = 50  # Number of images to capture per person

    while count < target_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access the camera.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128))
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            print(f"Captured image {count}/{target_images}")
        
        cv2.imshow("Capturing Faces", frame)
        key = cv2.waitKey(200)  # Wait 200ms before capturing the next frame
        if key & 0xFF == ord('q'):  # Allow manual quit
            print("Manual stop triggered.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing {count} images for {name}.")


# Step 3: Prepare Dataset
def load_dataset():
    images = []
    labels = []
    label_map = {}

    for idx, person in enumerate(os.listdir(DATASET_DIR)):
        person_path = os.path.join(DATASET_DIR, person)
        if os.path.isdir(person_path):
            label_map[idx] = person
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))
                images.append(img_to_array(img))
                labels.append(idx)
    
    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    return images, labels, label_map

# Step 4: Build and Train the Model
def train_model():
    images, labels, label_map = load_dataset()
    le = LabelEncoder()
    labels = to_categorical(le.fit_transform(labels))

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_map), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    model.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return label_map

# Step 5: Real-Time Face Recognition
def recognize_faces(label_map):
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128))
            face = img_to_array(face) / 255.0
            face = np.expand_dims(face, axis=0)

            predictions = model.predict(face)[0]
            confidence = np.max(predictions)
            label_idx = np.argmax(predictions)
            name = label_map[label_idx] if confidence > 0.6 else "Unknown"

            text = f"{name}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Driver Code
def main():
    print("1. Capture Faces")
    print("2. Train Model")
    print("3. Recognize Faces")
    choice = input("Enter your choice: ")

    if choice == "1":
        capture_faces()
    elif choice == "2":
        label_map = train_model()
        print("Label map:", label_map)
    elif choice == "3":
        _, _, label_map = load_dataset()
        recognize_faces(label_map)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
