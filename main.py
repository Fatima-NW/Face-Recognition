import face_recognition
import cv2
import os
import numpy as np
import chromadb
import time
import gradio as gr


THRESHOLD = 0.42


def average_person_embeddings(known_encodings, known_names, known_files):
    """ Compute average embeddings for each person """
    averaged_encodings = []
    averaged_names = []
    averaged_files = []

    unique_names = set(known_names)
    for name in unique_names:
        indices = [i for i, n in enumerate(known_names) if n == name]
        person_encs = [known_encodings[i] for i in indices]
        mean_enc = np.mean(person_encs, axis=0) 
        averaged_encodings.append(mean_enc)
        averaged_names.append(name)
        averaged_files.append(known_files[indices[0]])

    print(f"Averaged encodings for {len(unique_names)} unique people")
    return averaged_encodings, averaged_names, averaged_files


def init_face_db(known_dir="known_faces", db_path="face_db", collection_name="face_embeddings"):
    """ Initialize the face database using ChromaDB """
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    from_existing = load_existing_encodings(collection)
    known_encodings, known_names, known_files = add_new_faces_to_db(
        collection, known_dir, *from_existing
    )
    known_encodings, known_names, known_files = remove_deleted_faces_from_db(
        collection, known_dir, known_encodings, known_names, known_files
    )

    print(f"Total known encodings: {len(known_encodings)}")
    known_encodings, known_names, known_files = average_person_embeddings(known_encodings, known_names, known_files)
    return known_encodings, known_names


def load_existing_encodings(collection):
    """ Load known encodings and metadata from ChromaDB """
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


def add_new_faces_to_db(collection, known_dir, known_encodings, known_names, known_files):
    """ Add new face embeddings from known_faces directory to the ChromaDB """
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
        print(f"Stored {len(embeddings)} new encodings in DB")

    return known_encodings, known_names, known_files


def remove_deleted_faces_from_db(collection, known_dir, known_encodings, known_names, known_files):
    """ Remove deleted images from ChromaDB """
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
        print(f"Removed {len(to_delete)} encodings from DB")

        data = collection.get(include=["embeddings", "metadatas"])
        known_encodings = [np.array(e) for e in data["embeddings"]]
        known_names = [m["name"] for m in data["metadatas"]]
        known_files = [m["file"] for m in data["metadatas"]]

    return known_encodings, known_names, known_files


def recognize_faces(image, known_encodings, known_names, threshold=THRESHOLD, for_gradio=False):
    """ Detect and recognize faces in images """
   
    # Convert for recognition
    if for_gradio:
        image_rgb = image.copy()      # already RGB
        color_unknown = (255, 0, 0)   # RGB red
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color_unknown = (0, 0, 255)   # BGR red

    face_locations = face_recognition.face_locations(image_rgb)
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

    recognized_info = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            name = known_names[best_idx] if best_distance < threshold else "Unknown"
            confidence = 1 - best_distance
            recognized_info.append(f"{name} (distance={best_distance:.2f})")
        else:
            name = "Unknown"
            recognized_info.append("No known faces to compare")

        # Draw rectangles and labels
        colour = (0, 200, 0) if name != "Unknown" else color_unknown
        cv2.rectangle(image, (left, top), (right, bottom), colour, 2)

        if name != "Unknown":
            label_text = f"{name} ({confidence*100:.1f}%)"
        else:
            label_text = name
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = top - 10 if top - 10 > 10 else top + 10

        cv2.rectangle(image, (left, label_y - label_size[1]-4), (left + label_size[0] + 4, label_y + 4),  colour, cv2.FILLED)
        cv2.putText(image, label_text, (left + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image, recognized_info


def recognize_faces_opencv(test_image_path, known_encodings, known_names):
    """ Recognize faces using OpenCV window display """
    start_time = time.time()
    image = face_recognition.load_image_file(test_image_path)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_image, recognized_info = recognize_faces(image_bgr, known_encodings, known_names)
    runtime = time.time() - start_time
    for info in recognized_info:
        print(info)
    print(f"Runtime: {runtime:.2f} seconds")
    cv2.imshow("Face Recognition", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognize_faces_gradio(image, threshold, known_encodings, known_names):
    """ Recognize faces for Gradio interface """
    start_time = time.time()
    annotated_image, recognized_info = recognize_faces(image, known_encodings, known_names, threshold, for_gradio=True)
    runtime = time.time() - start_time
    recognized_text = "\n".join(recognized_info) + f"\nRuntime: {runtime:.2f} seconds"
    return annotated_image, recognized_text


def launch_gradio_ui(known_encodings, known_names):
    """ Launch Gradio interface """
    iface = gr.Interface(
        fn=lambda img, th: recognize_faces_gradio(img, th, known_encodings, known_names),
        inputs=[
            gr.Image(type="numpy"),
            gr.Slider(0.4, 1.0, value=THRESHOLD, step=0.01, label="Recognition Threshold") 
        ],
        outputs=[
            gr.Image(type="numpy"),
            gr.Textbox(lines=8, label="Recognition Results", interactive=False) 
        ],
        title="Face Recognition App"
    )
    iface.launch()


if __name__ == "__main__":
    known_encodings, known_names = init_face_db(known_dir="known_faces")

    # OpenCV test
    test_image_path = "test_images/group2.jpg"
    recognize_faces_opencv(test_image_path, known_encodings, known_names)

    # Launch Gradio UI
    launch_gradio_ui(known_encodings, known_names)