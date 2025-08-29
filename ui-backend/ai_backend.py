from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import uuid
import os
import cv2

# Load YOLOv3 model
model = YOLO("yolov3.pt")  # Make sure yolov3.pt is in the same folder

app = FastAPI(title="YOLOv3 AI Backend")

# Create folders if they don't exist
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "runs/detect"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Root endpoint to check server
@app.get("/")
def root():
    return {"message": "YOLOv3 FastAPI backend is running!"}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    # Save uploaded file with a random name
    file_ext = os.path.splitext(file.filename)[1]
    file_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detection
    results = model(file_path)

    # Save annotated image
    output_img_path = os.path.join(OUTPUT_FOLDER, file_name)
    annotated_img = results[0].plot()  # numpy array (BGR)
    cv2.imwrite(output_img_path, annotated_img)

    # Collect detection results
    detections = []
    for r in results:
        for obj in r.boxes.data.tolist():  # xyxy + confidence + class
            x1, y1, x2, y2, conf, cls = obj
            detections.append({
                "object": model.names[int(cls)],
                "confidence": float(conf),
                "bounding_box": [float(x1), float(y1), float(x2), float(y2)]
            })

    return JSONResponse(content={
        "detections": detections,
        "output_image": output_img_path
    })
