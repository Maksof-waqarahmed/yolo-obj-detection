from ultralytics import YOLO

# Train YOLOv8 model on custom dataset
model = YOLO("yolov8n.pt")  # Pretrained model
model.train(data="dataset.yml", epochs=50, imgsz=640)