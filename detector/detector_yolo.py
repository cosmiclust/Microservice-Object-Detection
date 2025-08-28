from ultralytics import YOLO
import json

# Load YOLOv3 model
model = YOLO("yolov3.pt")  # use the weights you downloaded

def run_detection(image_path, save_path="output.jpg"):
    results = model.predict(source=image_path, save=True)  # saves image with boxes

    detections = []
    for r in results:
        for obj in r.boxes.data.tolist():  # xyxy + confidence + class
            x1, y1, x2, y2, conf, cls = obj
            detections.append({
                "object": model.names[int(cls)],
                "confidence": float(conf),
                "bounding_box": [float(x1), float(y1), float(x2), float(y2)]
            })

    with open("output.json", "w") as f:
        json.dump({"detections": detections}, f, indent=4)

    return detections, save_path

if __name__ == "__main__":
    det, img = run_detection("yolov3/data/images/bus.jpg")
    print(det)
