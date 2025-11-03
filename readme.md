# Face Detection & Recognition App

A **Face Detection and Recognition** application built using **OpenCV** and **face_recognition**, capable of identifying and comparing human faces using multiple **similarity metrics** — including **Euclidean distance**, **Cosine similarity**, and **Manhattan distance**.

---

## Features

- 🧍 **Face Detection** — Automatically detects faces in real-time from images, videos, or webcam streams using OpenCV’s Haar cascades or HOG-based models.  
- 🧬 **Face Recognition** — Identifies or verifies faces using the `face_recognition` library (based on dlib’s deep learning face encodings).  
- 📏 **Similarity Metrics** — Calculates similarity between face encodings using:
  - **Euclidean Distance** (L2 norm)  
  - **Cosine Similarity**  
  - **Manhattan Distance** (L1 norm)  
- 📸 **Real-time Mode** — Supports live recognition from camera input.  
- 🗂️ **Database Support** — Easily add known faces to a local dataset for recognition.  

---

## 🧩 Technologies Used

| Library | Description |
|----------|--------------|
| **OpenCV** | For image capture, processing, and visualization. |
| **face_recognition** | For face detection and encoding generation. |
| **NumPy** | For vector operations and similarity calculations. |
| **dlib** | Backend model used by face_recognition for deep learning–based encoding. |

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/face-detection-app.git
   cd face-detection-app
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

##  How It Works

1. **Face Detection:**  
   The system detects faces in an image or video frame using OpenCV.

2. **Face Encoding:**  
   Each detected face is converted into a **128-dimensional embedding** using `face_recognition.face_encodings()`.

3. **Similarity Computation:**  
   The embeddings are compared using various distance metrics:
   ```python
   # Euclidean Distance
   euclidean_distance = np.linalg.norm(encoding1 - encoding2)

   # Cosine Similarity
   cosine_similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))

   # Manhattan Distance
   manhattan_distance = np.sum(np.abs(encoding1 - encoding2))
   ```

4. **Thresholding:**  
   If the distance/similarity is within a defined threshold, the faces are considered a match.

---

## 🧪 Usage

Run the app using:
```bash
python main.py
```

You can also provide an image as input:
```bash
python main.py --image path/to/image.jpg
```

For real-time webcam detection:
```bash
python main.py --camera
```

---

## 🖼️ Output Example

- Bounding boxes drawn around detected faces.  
- Names of recognized individuals displayed above faces.  
- Real-time similarity metrics shown in the console or on-screen.

---

## 📚 Folder Structure

```
face-detection-app/
│
├── data/
│   ├── known_faces/
│   └── unknown_faces/
│
├── main.py
├── requirements.txt
├── README.md
└── utils/
    ├── detection.py
    ├── recognition.py
    └── metrics.py
```

---

## 🧮 Example Output (Similarity Metrics)

| Metric | Value | Description |
|--------|--------|-------------|
| Euclidean | 0.42 | Smaller → More similar |
| Cosine | 0.89 | Closer to 1 → More similar |
| Manhattan | 5.2 | Smaller → More similar |

---

## 💡 Future Improvements

- Add support for deep CNN-based embeddings (e.g., FaceNet or ArcFace).  
- Integrate a GUI dashboard for easier interaction.  
- Optimize performance for real-time multi-face recognition.

---

## 🧑‍💻 Author

**Your Name**  
📧 [your.email@example.com](mailto:your.email@example.com)  
🔗 [GitHub Profile](https://github.com/yourusername)

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use and modify it.