# Face Recognition App

A python-based **face recognition** application that uses face_recognition, OpenCV, and ChromaDB for efficient face detection, encoding, and recognition — supporting both automated (OpenCV-based) and interactive (Gradio) modes.


## Features

- **Face Detection**: Automatically detects and localizes faces from images.

- **Face Recognition**: Identifies or verifies faces using pre-encoded known faces.  

- **Persistent Local Database**:  Uses ChromaDB to store and retrieve embeddings locally — no need to reprocess known faces every time.

- **Automatic Syncing**: Keeps the local database in sync with your `known_faces/` directory by adding and removing embeddings as files change.

- **Average Embedding per Person**: Combines multiple face images per person into one stable, averaged embedding for improved accuracy.

- **Similarity Metrics**: Calculates similarity between face encodings using **Euclidean Distance**. 

- **Dual Interface**:  
  - **OpenCV Mode:** Runs recognition on a sample image and saves result to `output/`.
  - **Gradio UI:** Interactive browser interface for real-time testing.

- **Performance Tracking**: Logs the total runtime for each recognition pass to help monitor system performance.


## Technologies Used

| Library | Purpose |
|----------|----------|
| **Python** | Core programming language |
| **OpenCV** | Image loading, display, and basic processing |
| **face_recognition** | Face detection and encoding (built on dlib) |
| **dlib** | Deep learning–based facial feature extraction engine used internally by face_recognition |
| **NumPy** | Numerical operations and vector similarity calculations |
| **ChromaDB** | Local vector database to store and query embeddings efficiently |
| **Gradio** | Web interface for live testing and visualization |
| **time** | Performance measurement utilities |


## Workflow

1. **Database Initialization**    
   - Loads existing embeddings from ChromaDB.  
   - Adds new faces from `known_faces/`.  
   - Cleans up deleted or outdated entries automatically.

2. **Embedding Averaging**      
   Multiple images of each person are averaged to create a single, robust embedding.

3. **Recognition Process**              
   - Detects and encodes faces in a new image.  
   - Compares each encoding against all known embeddings using **Euclidean distance**.  
   - Matches faces with distance below the defined threshold.

4. **Runtime Logging**              
   Total recognition time (detection + matching) is displayed in the console or Gradio UI.


## Project Structure

```
face-detection-app/
│
├── main.py                 # Core script
├── face_db/                # ChromaDB persistent database files
├── known_faces/            # Folder containing known people (subfolders by name)
│   ├── Name1/
│   │   ├── img1.jpg
│   │   ├── img2.png
│   └── Name2/
│       └── photo.jpg
├── test_images/            # Test images for recognition
│   ├── test1.jpeg
│   └── test2.jpg
├── output/                 # Automatically stores annotated images from OpenCV
├── venv/                   # Virtual environment
├── Dockerfile              # Docker
├── docker-compose.yml 
├── .dockerignore 
├── .env                    # Configurable runtime parameters
├── .gitignore
├── requirements.txt        # Dependencies
└── README.md
```

## Setup and Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Fatima-NW/Face-Recognition.git
   ```

2. **Environment Configuration**                
   You can adjust these values without modifying the code.
   ```bash
   THRESHOLD=0.42
   TEST_IMAGE=group.png     # Place this image in test_images/ folder
   ```

Next, choose either option A or B depending on whether you want to run the app manually (classic Python setup) or inside Docker.

### A) Classic Python Setup

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Project**

   ```bash
   python main.py
   ```

   The script will:
   - Initialize and sync the ChromaDB face database.
   - Recognize faces in a sample test image using OpenCV.
   - Launch an interactive Gradio web interface.


### B) Docker Setup

3. **Build and start the container**
   ```bash
   sudo docker compose up --build

   #After the first build, you can start containers normally without rebuilding:
   sudo docker compose up
   ```

4. **Access the web UI**                             
   Visit: http://127.0.0.1:7860/

5. **Check Logs (optional)**
   ```bash
   sudo docker logs face_recognition_app
   ```

6. **Stop container**
   ```bash
   sudo docker compose down
   ```


## Interfaces

### OpenCV Mode     
- Runs automatically at startup **if a test image is provided** in the `.env` file.  
- Displays results in the console, including:
  - Detected person names  
  - Euclidean distance (similarity) values  
  - Total recognition runtime  
- Saves the annotated image in the `output/` directory:  
  - **Green boxes** for recognized faces 
  - **Red boxes** for unknown faces  
  - **Name + confidence score** above each detected face  

### Gradio Mode    
- Launches an interactive web UI at [http://127.0.0.1:7860](http://127.0.0.1:7860).  
- Allows users to:
  - Upload an image  
  - Adjust recognition threshold dynamically  
- Displays the **same annotated results** as OpenCV (green/red boxes with names and confidence scores).  
- Also shows the recognition results below the image with names, distances, and runtime.


## Notes

- Ensure the `known_faces/` directory has subfolders named after each person.
- Set the threshold value and provide test image (also place it in the test_images folder) in .env file before running.
- A **lower distance value** indicates a **stronger match** (higher confidence), since similarity is calculated using Euclidean distance.
- Use high-quality, front-facing images for best results.
- Logs from the OpenCV mode appear in the console output.

