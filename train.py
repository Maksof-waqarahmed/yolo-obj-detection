# from ultralytics import YOLO

# # Train YOLOv8 model on custom dataset
# model = YOLO("yolov8n.pt")  # Pretrained model
# model.train(data="dataset.yml", epochs=50, imgsz=640)


from ultralytics import YOLO

# Train YOLOv8 model with better parameters
model = YOLO("yolov8n.pt")  

model.train(
    data="dataset.yml",
    epochs=30,  # Increase epochs for better learning
    imgsz=640,
    batch=16,  # Adjust batch size based on your system memory
    device="cpu",  # Use 'cuda' if you have GPU
    optimizer="AdamW",  # Better optimizer for small datasets
    lr0=0.001,  # Reduce initial learning rate to avoid overfitting
    weight_decay=5e-4,  # Regularization
    augment=True  # Apply more augmentations
)
