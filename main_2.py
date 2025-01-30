
from ultralytics import YOLO
from PIL import Image
import cv2

# Load your YOLO model (update the path based on training results)
model = YOLO("runs/detect/train1/weights/best.pt")
image_path = "test3.jpg"
image = cv2.imread(image_path)
# image = cv2.resize(image, (640, 640))
# 🔹 Object Detection perform karo
results = model(image,show=True)
detected_class = results[0].names[int(results[0].boxes[0].cls)]
print(detected_class)