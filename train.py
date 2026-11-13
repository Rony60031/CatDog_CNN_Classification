import os
import time
import csv
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
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
LEARNING_RATE = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 4. Load Training Dataset
# =====================================

train_dataset = CatDogDataset(
    TRAIN_DIR,
    transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print("Training images:", len(train_dataset))
print("Batch size:", BATCH_SIZE)
print("Total batches:", len(train_loader))


# =====================================
# 5. Create CNN Model
# =====================================

model = Model_CatDog().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9
)


# =====================================
# 6. Training
# =====================================

history = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    print(f"\nStarting Epoch {epoch + 1}/{EPOCHS}")

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

        if (batch_idx + 1) % 500 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)}"
            )

    torch.cuda.synchronize()

    epoch_loss = total_loss / total
    epoch_accuracy = 100 * correct / total

    elapsed = time.time() - start_time

    history.append({
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "accuracy": epoch_accuracy
    })

    print("\nEpoch:", epoch + 1)
    print("Training Loss:", round(epoch_loss, 4))
    print("Training Accuracy:", round(epoch_accuracy, 2), "%")
    print("Time:", round(elapsed / 60, 2), "minutes")

    # Save checkpoint after every epoch
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history
    }, "model/catdog_checkpoint.pth")


# =====================================
# 7. Save Training History
# =====================================

with open(
    "results/training_history.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["epoch", "loss", "accuracy"]
    )

    writer.writeheader()
    writer.writerows(history)


print("\n=================================")
print("TRAINING COMPLETED!")
print("Model saved successfully!")
print("Training history saved!")
print("=================================")