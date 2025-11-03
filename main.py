import face_recognition
import cv2
import os
import numpy as np
import chromadb
import time
import gradio as gr


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
        print(f"Loaded {len(known_encodings)} encodings from ChromaDB")
    else:
        print("No existing data found in DB")

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
        print(f"Stored {len(embeddings)} new encodings in ChromaDB")

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
        print(f"Removed {len(to_delete)} embeddings from DB")

        data = collection.get(include=["embeddings", "metadatas"])
        known_encodings = [np.array(e) for e in data["embeddings"]]
        known_names = [m["name"] for m in data["metadatas"]]
        known_files = [m["file"] for m in data["metadatas"]]

    return known_encodings, known_names, known_files


# Detect faces
def recognize_faces(image, known_encodings, known_names, known_files, threshold=0.46):
    """Recognize faces from Gradio-uploaded image."""
    start_time = time.time()
    test_image = image
    # test_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    recognized_info = []

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
            recognized_info.append(f"{name} (distance={best_distance:.2f}, matched file={matched_file})")
        else:
            name = "Unknown"
            recognized_info.append("No known faces to compare")

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        text_y = bottom + 25
        cv2.rectangle(image, (left, bottom), (right, text_y), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, name, (left + 5, bottom + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    runtime = time.time() - start_time
    recognized_text = "\n".join(recognized_info) + f"\nRuntime: {runtime:.2f} seconds"

    return image, recognized_text


# Launch Gradio UI
def launch_gradio_ui():
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

    iface = gr.Interface(
        fn=lambda img: recognize_faces(img, known_encodings, known_names, known_files),
        inputs=gr.Image(type="numpy"),
        outputs=[gr.Image(type="numpy"), gr.Text()],
        title="Face Recognition App"
    )
    iface.launch()


if __name__ == "__main__":
    launch_gradio_ui()