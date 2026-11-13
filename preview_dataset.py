import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Dataset location
DATASET_PATH = r"D:\data\catdog"

TRAIN_PATH = os.path.join(DATASET_PATH, "training_set")

# Find image files
files = [
    f for f in os.listdir(TRAIN_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Select 8 random images
random.seed(42)
samples = random.sample(files, 8)

# Display images
plt.figure(figsize=(12, 8))

for i, filename in enumerate(samples):

    image_path = os.path.join(TRAIN_PATH, filename)

    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Get label from filename
    label = "Cat" if "cat" in filename.lower() else "Dog"

    plt.subplot(2, 4, i + 1)
    plt.imshow(image)
    plt.title(label)
    plt.axis("off")

plt.suptitle("Cat and Dog Dataset - Sample Images")
plt.tight_layout()
plt.show()