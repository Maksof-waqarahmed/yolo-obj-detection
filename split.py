import os
import shutil
import random

# Set paths
dataset_path = "dataset/"  # Folder jisme saari images hain
train_path = "train/"
val_path = "val/"

# Create train & val directories if not exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Get all images
all_images = [img for img in os.listdir(dataset_path) if img.endswith((".jpg", ".png"))]
random.shuffle(all_images)

# Split 80% train, 20% val
split_index = int(len(all_images) * 0.8)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Move images
for img in train_images:
    shutil.move(os.path.join(dataset_path, img), os.path.join(train_path, img))

for img in val_images:
    shutil.move(os.path.join(dataset_path, img), os.path.join(val_path, img))

print("✅ Data Split Complete: Train =", len(train_images), " Val =", len(val_images))
