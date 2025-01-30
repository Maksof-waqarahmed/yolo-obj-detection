from ultralytics import YOLO

model = YOLO("runs/detect/train10/weights/best.pt")
results = model.predict("test3.jpg", conf=0.5)  # set confidence threshold to 0.5
print(results)
# results.show()