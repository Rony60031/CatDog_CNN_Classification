import os
import time
import csv
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from model import Model_CatDog


# =====================================
# 1. Configuration
# =====================================

DATASET_PATH = r"D:\data\catdog"
TRAIN_DIR = os.path.join(DATASET_PATH, "training_set")

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 0.0001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================
# 2. Image Transformation
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =====================================
# 3. Dataset Class
# =====================================

class CatDogDataset(Dataset):

    def __init__(self, directory, transform=None):

        self.directory = directory
        self.transform = transform

        self.files = [
            f for f in os.listdir(directory)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ]

        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        image_path = os.path.join(
            self.directory, filename
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if "cat" in filename.lower():
            label = 0
        elif "dog" in filename.lower():
            label = 1
        else:
            raise ValueError(filename)

        return image, label


# =====================================
# 4. Load Dataset
# =====================================

full_dataset = CatDogDataset(
    TRAIN_DIR,
    transform
)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =====================================
# 5. Create Model
# =====================================

model = Model_CatDog().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================
# 6. Training and Validation
# =====================================

history = []
best_val_loss = float("inf")

for epoch in range(EPOCHS):

    start_time = time.time()

    # -----------------------------
    # Training
    # -----------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= train_total

    train_accuracy = (
        100 * train_correct / train_total
    )

    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total

    val_accuracy = (
        100 * val_correct / val_total
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start_time

    # -----------------------------
    # Display Results
    # -----------------------------

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print("Training Loss:", round(train_loss, 4))
    print("Training Accuracy:", round(train_accuracy, 2), "%")

    print("Validation Loss:", round(val_loss, 4))
    print("Validation Accuracy:", round(val_accuracy, 2), "%")

    print("Time:", round(elapsed / 60, 2), "minutes")

    # -----------------------------
    # Save History
    # -----------------------------

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    # -----------------------------
    # Save Best Model
    # -----------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "model/best_catdog_model.pth"
        )

        print("Best model saved!")

    # Save latest checkpoint
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history
    }, "model/validation_checkpoint.pth")


# =====================================
# 7. Save Training History
# =====================================

with open(
    "results/validation_history.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(history)


print("\n=================================")
print("TRAINING AND VALIDATION COMPLETED!")
print("Best model saved successfully!")
print("History saved successfully!")
print("=================================")