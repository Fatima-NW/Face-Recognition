import face_recognition
import cv2
import os
import numpy as np
import chromadb
import time


# Load known encodings
def load_existing_encodings(collection):
    """Load known encodings and metadata from ChromaDB."""
    known_encodings, known_names, known_files = [], [], []
    data = collection.get(include=["embeddings", "metadatas"])

    if len(data["embeddings"]) > 0 and len(data["metadatas"]) > 0:
        for emb, meta in zip(data["embeddings"], data["metadatas"]):
            known_encodings.append(np.array(emb))
            known_names.append(meta["name"])
            known_files.append(meta["file"])
        print(f"Loaded {len(known_encodings)} encodings from ChromaDB.")
    else:
        print("No existing data found in DB.")

    return known_encodings, known_names, known_files


# Add new images to DB
def add_new_faces_to_db(collection, known_dir, known_encodings, known_names, known_files):
    """Add new faces from known_faces directory to the database."""
    ids, embeddings, metadatas = [], [], []
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    for person_name in os.listdir(known_dir):
        person_path = os.path.join(known_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        for file in os.listdir(person_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_id = f"{person_name}_{file}"
            if img_id in existing_ids:
                continue

            image_path = os.path.join(person_path, file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                emb = np.array(encodings[0], dtype=np.float32).tolist()
                embeddings.append(emb)
                ids.append(img_id)
                metadatas.append({
                    "name": person_name,
                    "file": file,
                    "path": image_path
                })
                known_encodings.append(np.array(emb))
                known_names.append(person_name)
                known_files.append(file)
                print(f"Added new face: {file} ({person_name})")

    if embeddings:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"Stored {len(embeddings)} new encodings in ChromaDB.")

    return known_encodings, known_names, known_files


# Remove deleted images from DB
def remove_deleted_faces_from_db(collection, known_dir, known_encodings, known_names, known_files):
    """Remove deleted images from the database and refresh local lists."""
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    current_ids = set()

    for person_name in os.listdir(known_dir):
        person_path = os.path.join(known_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        for file in os.listdir(person_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_id = f"{person_name}_{file}"
                current_ids.add(img_id)

    to_delete = list(existing_ids - current_ids)
    if to_delete:
        collection.delete(ids=to_delete)
        print(f"Removed {len(to_delete)} embeddings from DB (no longer in folder).")

        data = collection.get(include=["embeddings", "metadatas"])
        known_encodings = [np.array(e) for e in data["embeddings"]]
        known_names = [m["name"] for m in data["metadatas"]]
        known_files = [m["file"] for m in data["metadatas"]]

    return known_encodings, known_names, known_files


# Detect faces
def recognize_faces(test_image_path, known_encodings, known_names, known_files, threshold=0.45):
    """Recognize faces in the test image."""
    test_image = face_recognition.load_image_file(test_image_path)
    test_image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    print(f"Detected {len(face_encodings)} faces in test image.")

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]

            if best_distance < threshold:
                name = known_names[best_match_index]
            else:
                name = "Unknown"

            matched_file = known_files[best_match_index]
            print(f"Found: {name} (distance={best_distance:.2f}, matched file={matched_file})")
        else:
            name = "Unknown"
            print("No known faces to compare.")

        # Draw results
        cv2.rectangle(test_image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        text_y = bottom + 25
        cv2.rectangle(test_image_bgr, (left, bottom), (right, text_y), (0, 255, 0), cv2.FILLED)
        cv2.putText(test_image_bgr, name, (left + 5, bottom + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Face Recognition", test_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    start_time = time.time()

    chroma_client = chromadb.PersistentClient(path="face_db")
    collection = chroma_client.get_or_create_collection(name="face_embeddings")

    known_dir = "known_faces"

    known_encodings, known_names, known_files = load_existing_encodings(collection)
    known_encodings, known_names, known_files = add_new_faces_to_db(
        collection, known_dir, known_encodings, known_names, known_files
    )
    known_encodings, known_names, known_files = remove_deleted_faces_from_db(
        collection, known_dir, known_encodings, known_names, known_files
    )

    print(f"Total known encodings: {len(known_encodings)}")

    test_image_path = "test_images/group2.jpg"
    recognize_faces(test_image_path, known_encodings, known_names, known_files)

    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
