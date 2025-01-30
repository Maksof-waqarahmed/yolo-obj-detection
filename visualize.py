import cv2
import os

IMAGES_FOLDER = "dataset_sku/images/test"
LABELS_FOLDER = "dataset_sku/labels/test"

# Classes list (Same as dataset.yaml)
CLASSES = ["Coca Cola", "Sprite", "Pepsi", "Mountain Dew", "7UP", "Fanta"]

def draw_boxes(image_path, label_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, width, height = map(float, data[1:])

            # Convert YOLO format to pixels
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, CLASSES[class_id], (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Labeled Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run visualization on one image
test_image = "10a743af-bdaf-4486-be1c-41a269e7f8b1.jpg"  # Change this to an existing image name
draw_boxes(f"{IMAGES_FOLDER}/{test_image}", f"{LABELS_FOLDER}/{test_image.replace('.jpg', '.txt')}")
