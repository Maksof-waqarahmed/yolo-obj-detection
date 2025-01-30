import cv2
from ultralytics import YOLO
from collections import defaultdict

# ✅ Train kiya hua model load karo
model = YOLO("best.pt")  

# ✅ Image load karo
image_path = "test1.jpg"
image = cv2.imread(image_path)

# ✅ Check image resolution
print("Image Shape:", image.shape)

# ✅ Object Detection perform karo (low confidence)
results = model(image, conf=0.2)  

# ✅ Stock count dictionary
stock_counts = defaultdict(int)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        label = model.names[int(box.cls[0])]  
        confidence = box.conf[0].item()  

        stock_counts[label] += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ✅ Print detected items
print("\n📦 Detected Stock Counts:")
for item, count in stock_counts.items():
    print(f"{item}: {count}")

# ✅ Show image
cv2.imshow("Stock Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
