Microservice Object Detection

This project demonstrates a microservice-based approach to object detection using YOLOv3 with FastAPI as the backend. The repository is divided into two main components:

detector/ – Handles the object detection logic using YOLOv3.

ui-backend/ – A FastAPI backend service that exposes APIs for running detections and returning results.

The microservice design ensures modularity, making it easier to maintain, extend, and deploy the system in containerized environments using Docker.

Repository Structure
detector/
    ├── output_images/        # Contains model-annotated output images
    ├── sample_images/        # Sample input images for testing
    ├── detect_ai.py          # Main detection script using YOLOv3
    ├── Dockerfile            # Dockerfile for building the detection service
    ├── Link to YoloV3 weights.md  # Instructions to download YOLOv3 weights
    └── requirement.txt       # Python dependencies for detection

ui-backend/
    ├── ai_backend.py         # FastAPI backend exposing detection APIs
    ├── Dockerfile            # Dockerfile for backend service
    └── requirements.txt      # Backend dependencies

Setup Instructions
1. Clone the Repository
git clone https://github.com/cosmiclust/Microservice-Object-Detection.git
cd Microservice-Object-Detection

2. Download YOLOv3 Weights

The YOLOv3 model weights are required for detection. Follow the instructions provided in
detector/Link to YoloV3 weights.md to place the weights file (yolov3.pt) in the appropriate directory.

3. Install Dependencies
Option A: Run locally with Python

Make sure you have Python 3.11 installed.

For detector/:

cd detector
pip install -r requirement.txt


For ui-backend/:

cd ../ui-backend
pip install -r requirements.txt

Option B: Run with Docker

Each service comes with its own Dockerfile. Build and run the containers as follows:

For detector/:

cd detector
docker build -t detector-service .
docker run -p 8001:8001 detector-service


For ui-backend/:

cd ../ui-backend
docker build -t ui-backend-service .
docker run -p 8000:8000 ui-backend-service

Running the Services

Start both services (detector and UI-backend).

Once the containers are running, the UI-backend service should be accessible at:

http://127.0.0.1:8000/docs


This will open the FastAPI interactive documentation (Swagger UI).

Use the /detect endpoint to upload an image and receive:

A JSON response with detected objects, confidence scores, and bounding boxes.

A processed image with bounding boxes drawn around detected objects.

Example Workflow

Upload a sample image (from detector/sample_images/) through the /detect endpoint.

The backend processes the image and saves an annotated output in detector/output_images/.

The API returns JSON results similar to:

{
  "detections": [
    {
      "object": "person",
      "confidence": 0.92,
      "bounding_box": [34.5, 56.7, 200.1, 350.4]
    }
  ],
  "output_image": "runs/detect/83c89f20-dfd3-4d7a-8dfd-8f057e156fa2.jpg"
}


The output image path can be used to view the annotated result.

