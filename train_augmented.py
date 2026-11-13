
import os
import csv
import time
import random
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

from model import Model_CatDog


# =====================================
# 1. Configuration
# =====================================

SEED = 42
BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 0.0001
PATIENCE = 4

BASE_DIR = r"D:\data\catdog"
TRAIN_DIR = os.path.join(BASE_DIR, "training_set")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Device:", device)


# =====================================
# 2. Training Transformation
# =====================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# =====================================
# 3. Validation Transformation
# =====================================

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# =====================================
# 4. Dataset
# =====================================

class CatDogDataset(Dataset):

    def __init__(self, directory, transform=None):

        self.directory = directory
        self.transform = transform

        self.files = sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        path = os.path.join(
            self.directory, filename
        )

        with Image.open(path) as img:
            image = img.convert("RGB")

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
# 5. Stratified Train/Validation Split
# =====================================

base_dataset = CatDogDataset(TRAIN_DIR)

cat_indices = [
    i for i, f in enumerate(base_dataset.files)
    if "cat" in f.lower()
]

dog_indices = [
    i for i, f in enumerate(base_dataset.files)
    if "dog" in f.lower()
]

rng = random.Random(SEED)

rng.shuffle(cat_indices)
rng.shuffle(dog_indices)

def split_class(indices):
    n_train = int(0.9 * len(indices))
    return indices[:n_train], indices[n_train:]

cat_train, cat_val = split_class(cat_indices)
dog_train, dog_val = split_class(dog_indices)

train_indices = cat_train + dog_train
val_indices = cat_val + dog_val

rng.shuffle(train_indices)
rng.shuffle(val_indices)

train_base = CatDogDataset(
    TRAIN_DIR, train_transform
)

val_base = CatDogDataset(
    TRAIN_DIR, val_transform
)

train_dataset = Subset(
    train_base, train_indices
)

val_dataset = Subset(
    val_base, val_indices
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

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Training Cats:", len(cat_train))
print("Training Dogs:", len(dog_train))


# =====================================
# 6. Model and Optimizer
# =====================================

model = Model_CatDog().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================
# 7. Training
# =====================================

history = []
best_val_loss = float("inf")
best_epoch = 0
bad_epochs = 0

for epoch in range(EPOCHS):

    start_time = time.time()

    # Training
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = 100 * correct / total

    # Validation
    model.eval()

    val_total_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_total_loss / val_total
    val_acc = 100 * val_correct / val_total

    torch.cuda.synchronize()

    elapsed = time.time() - start_time

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("Training Loss:", round(train_loss, 4))
    print("Training Accuracy:", round(train_acc, 2), "%")
    print("Validation Loss:", round(val_loss, 4))
    print("Validation Accuracy:", round(val_acc, 2), "%")
    print("Time:", round(elapsed / 60, 2), "minutes")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        bad_epochs = 0

        torch.save(
            model.state_dict(),
            "model/best_augmented_model.pth"
        )

        print("Best augmented model saved!")

    else:
        bad_epochs += 1

    # Save progress
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }, "model/augmented_checkpoint.pth")

    # Early stopping
    if bad_epochs >= PATIENCE:
        print("\nEarly stopping activated.")
        break


# =====================================
# 8. Save History
# =====================================

with open(
    "results/augmented_history.csv",
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

print("\n================================")
print("AUGMENTED TRAINING COMPLETED!")
print("Best Epoch:", best_epoch)
print("Best Validation Loss:", round(best_val_loss, 4))
print("================================")
