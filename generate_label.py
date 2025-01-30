import cv2
import os

# ✅ Input & Output Folder Paths
IMAGES_FOLDER = "dataset_2/test/images"
LABELS_FOLDER = "dataset_2/test/labels"

# ✅ Ensure labels directory exists
os.makedirs(LABELS_FOLDER, exist_ok=True)

# ✅ Variables for drawing bounding boxes
drawing = False
ix, iy, fx, fy = -1, -1, -1, -1
boxes = []
current_image = None
current_image_name = ""

# ✅ Classes list (Change as per your dataset)
CLASSES = ["Hello"]
current_class = 0  # Default class index

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, boxes, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
            img_copy = current_image.copy()
            cv2.rectangle(img_copy, (ix, iy), (fx, fy), (0, 255, 0), 2)
            cv2.imshow("Image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        cv2.rectangle(current_image, (ix, iy), (fx, fy), (0, 255, 0), 2)
        
        # Normalize values (YOLO format)
        h, w, _ = current_image.shape
        x_center = (ix + fx) / 2 / w
        y_center = (iy + fy) / 2 / h
        width = abs(fx - ix) / w
        height = abs(fy - iy) / h

        boxes.append((current_class, x_center, y_center, width, height))
        print(f"Added: Class {CLASSES[current_class]}, Box: {boxes[-1]}")

def save_labels():
    if not boxes:
        return
    label_path = os.path.join(LABELS_FOLDER, f"{current_image_name}.txt")
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
    print(f"Saved labels to {label_path}")

def retry_last_box():
    """Reset the last bounding box for retry."""
    if boxes:
        boxes.pop()  # Remove the last box from the list
        print("Retrying the last box...")
    cv2.imshow("Image", current_image)

def main():
    global current_image, current_image_name, boxes, current_class

    image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.endswith((".jpg", ".png"))]
    
    for image_name in image_files:
        image_path = os.path.join(IMAGES_FOLDER, image_name)
        current_image = cv2.imread(image_path)
        current_image_name = os.path.splitext(image_name)[0]
        boxes = []

        cv2.imshow("Image", current_image)
        cv2.setMouseCallback("Image", draw_rectangle)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):  # Save and Next
                save_labels()
                break
            elif key == ord("q"):  # Quit
                save_labels()
                cv2.destroyAllWindows()
                return
            elif key == ord("c"):  # Change Class
                current_class = (current_class + 1) % len(CLASSES)
                print(f"Selected Class: {CLASSES[current_class]}")
            elif key == ord("r"):  # Retry last bounding box
                retry_last_box()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
