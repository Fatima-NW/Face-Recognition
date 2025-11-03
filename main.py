import face_recognition
import cv2
import os
import time
start_time = time.time()

# Step 1: Load known faces
known_encodings = []
known_names = []
known_files= []

known_dir = "known_faces"
for person_name in os.listdir(known_dir):
    person_path = os.path.join(known_dir, person_name)
    if os.path.isdir(person_path):
        for file in os.listdir(person_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_path, file)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    known_files.append(file)

print(f"Loaded {len(known_encodings)} face images from {len(set(known_names))} people.")

# Step 2: Load test image
test_image_path = "test_images/group2.jpg"
test_image = face_recognition.load_image_file(test_image_path)
test_image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

# Step 3: Detect faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

print(f"Detected {len(face_encodings)} faces in test image.")

# Step 4: Compare detected faces with known faces
threshold = 0.45  # distance threshold for match

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    if known_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = distances.argmin()
        best_distance = distances[best_match_index]
        if best_distance < threshold:
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        matched_file = known_files[best_match_index]  # NEW
        print(f"Found: {name} (distance={best_distance:.2f}, matched file={matched_file})")

    else:
        name = "Unknown"
        print("No known faces to compare.")

    # Draw bounding box and label
    cv2.rectangle(test_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

    
    text_y = bottom + 25
    cv2.rectangle(test_image_bgr, (left, bottom), (right, text_y), (0, 255, 0), cv2.FILLED)
    cv2.putText(test_image_bgr, name, (left + 5, bottom + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Step 5: Show image
cv2.imshow("Face Recognition", test_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

end_time = time.time()
print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")