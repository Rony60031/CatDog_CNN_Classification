
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
EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 31

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
# 2. Transformations
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

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# =====================================
# 3. Dataset
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
        path = os.path.join(self.directory, filename)

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
# 4. Stratified Split
# =====================================

base_dataset = CatDogDataset(TRAIN_DIR)

cats = [
    i for i, f in enumerate(base_dataset.files)
    if "cat" in f.lower()
]

dogs = [
    i for i, f in enumerate(base_dataset.files)
    if "dog" in f.lower()
]

rng = random.Random(SEED)

rng.shuffle(cats)
rng.shuffle(dogs)

def split_class(indices):
    n = int(0.9 * len(indices))
    return indices[:n], indices[n:]

cat_train, cat_val = split_class(cats)
dog_train, dog_val = split_class(dogs)

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

train_dataset = Subset(train_base, train_indices)
val_dataset = Subset(val_base, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =====================================
# 5. Original CNN Architecture
# =====================================

model = Model_CatDog().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=0.000001
)


# =====================================
# 6. Training
# =====================================

history = []

best_val_loss = float("inf")
best_epoch = 0
bad_epochs = 0

for epoch in range(EPOCHS):

    start = time.time()

    # Training
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

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
    train_acc = 100 * train_correct / train_total

    # Validation
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = 100 * val_correct / val_total

    scheduler.step(val_loss)

    torch.cuda.synchronize()

    elapsed = time.time() - start

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("Training Loss:", round(train_loss, 4))
    print("Training Accuracy:", round(train_acc, 2), "%")
    print("Validation Loss:", round(val_loss, 4))
    print("Validation Accuracy:", round(val_acc, 2), "%")
    print("Learning Rate:", current_lr)
    print("Time:", round(elapsed / 60, 2), "minutes")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": current_lr
    })

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        bad_epochs = 0

        torch.save(
            model.state_dict(),
            "model/best_optimized_model.pth"
        )

        print("Best model saved!")

    else:
        bad_epochs += 1

    # Save checkpoint
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "bad_epochs": bad_epochs
    }, "model/optimized_checkpoint.pth")

    if bad_epochs >= PATIENCE:
        print("\nEarly stopping activated!")
        break


# =====================================
# 7. Save History
# =====================================

with open(
    "results/optimized_history.csv",
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
            "val_accuracy",
            "learning_rate"
        ]
    )

    writer.writeheader()
    writer.writerows(history)

print("\n================================")
print("OPTIMIZED CNN TRAINING COMPLETED")
print("Best Epoch:", best_epoch)
print("Best Validation Loss:", round(best_val_loss, 4))
print("================================")
