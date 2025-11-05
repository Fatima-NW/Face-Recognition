# Face Recognition App

A python-based **face recognition** application that uses face_recognition, OpenCV, and ChromaDB for efficient face detection, encoding, and recognition — with both CLI (OpenCV) and Gradio web UI interfaces.


## Features

- **Face Detection**: Automatically detects and localizes faces from images.

- **Face Recognition**: Identifies or verifies faces using pre-encoded known faces.  

- **Persistent Local Database**:  Uses ChromaDB to store and retrieve embeddings locally — no need to reprocess known faces every time.

- **Automatic Syncing**: Keeps the local database in sync with your `known_faces/` directory by adding and removing embeddings as files change.

- **Average Embedding per Person**: Combines multiple face images per person into one stable, averaged embedding for improved accuracy.

- **Similarity Metrics**: Calculates similarity between face encodings using **Euclidean Distance**. 

- **Configurable Threshold**: Tune the similarity threshold (`THRESHOLD = 0.42` by default) to control recognition confidence.

- **Dual Interface**:  
  - **OpenCV Mode:** Runs recognition on sample images directly.
  - **Gradio UI:** Interactive browser interface for real-time testing.

- **Performance Tracking**: Logs the total runtime for each recognition pass to help monitor system performance.


## Technologies Used

| Library | Purpose |
|----------|----------|
| **Python** | Core programming language. |
| **OpenCV** | Image loading, display, and basic processing. |
| **face_recognition** | Face detection and encoding (built on dlib). |
| **dlib** | Deep learning–based facial feature extraction engine used internally by face_recognition. |
| **NumPy** | Numerical operations and vector similarity calculations. |
| **ChromaDB** | Local vector database to store and query embeddings efficiently. |
| **Gradio** | Web interface for live testing and visualization. |
| **time** | Performance measurement utilities. |


## Workflow

1. **Database Initialization**    
   - Loads existing embeddings from ChromaDB.  
   - Adds new faces from `known_faces/`.  
   - Cleans up deleted or outdated entries automatically.

2. **Embedding Averaging**      
   Multiple images of each person are averaged to create a single, robust embedding.

3. **Recognition Process**              
   - Detects and encodes faces in a new image.  
   - Compares each encoding against all known embeddings using Euclidean distance.  
   - Matches faces with distance below the defined threshold.

4. **Runtime Logging**              
   Total recognition time (detection + matching) is displayed in the console or Gradio UI.


## Project Structure

```
face-detection-app/
│
├── main.py                 # Core script
├── known_faces/            # Folder containing known people (subfolders by name)
│   ├── person1/
│   │   ├── img1.jpg
│   │   ├── img2.png
│   └── person2/
│       └── photo.jpg
├── test_images/            # Test images for recognition
│   ├── test1.jpeg
│   └── test2.jpg
├── face_db/                # ChromaDB persistent database files
├── requirements.txt        # Dependencies
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Fatima-NW/Face-Recognition.git
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

In your virtual environment run:

   ```bash
   python main.py
   ```

The script will:
- Initialize and sync the ChromaDB face database.
- Recognize faces in a sample test image using OpenCV.
- Launch an interactive Gradio web interface.


## Interfaces

### OpenCV Mode     
Displays a test image (test_images/group2.jpg) annotated with recognized faces.

### Gradio Mode    
Launches a web UI at http://127.0.0.1:7860, where you can:
- Upload an image.
- Adjust recognition threshold.
- View detected faces, labels, and confidence scores in real time.
