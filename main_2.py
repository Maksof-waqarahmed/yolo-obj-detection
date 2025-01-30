
from ultralytics import YOLO
# from PIL import Image
import cv2

# Load your YOLO model (update the path based on training results)
model = YOLO("best.pt")
image_path = "fanta.jpg"
image = cv2.imread(image_path)
# image = cv2.resize(image, (640, 640))

# Perform object detection
results = model.predict(image, show=True)  # 🔹 `predict()` use karo, direct model() nahi!

# # 🔹 Object Detection perform karo
# results = model(image,show=True)
# print(results)
for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(f"Class: {model.names[class_id]}, Confidence: {confidence:.2f}")
# detected_class = results[0].names[int(results[0].boxes[0].cls)]
# print(detected_class)


