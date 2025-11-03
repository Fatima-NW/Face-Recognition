import face_recognition
import cv2
import os

# Step 1: Load known faces
known_encodings = []
known_names = []

known_dir = "known_faces"
for file in os.listdir(known_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = face_recognition.load_image_file(os.path.join(known_dir, file))
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0].strip())  

print(f"Loaded {len(known_encodings)} known faces.")


test_image_path = "test_images/ind.jpeg"   
test_image = face_recognition.load_image_file(test_image_path)
test_image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)


face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

print(f"Detected {len(face_encodings)} faces in test image.")

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Calculate distances to all known faces
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    
    if len(face_distances) > 0:
        best_match_index = face_distances.argmin() 
        best_distance = face_distances[best_match_index]
        threshold = 0.45  

        if best_distance < threshold:
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        print(f"Found: {name} (distance={best_distance:.2f})")
    else:
        name = "Unknown"
        print("No known faces to compare.")

    
    cv2.rectangle(test_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

    
    text_y = bottom + 25
    cv2.rectangle(test_image_bgr, (left, bottom), (right, text_y), (0, 255, 0), cv2.FILLED)
    cv2.putText(test_image_bgr, name, (left + 5, bottom + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

cv2.imshow("Face Recognition", test_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()



