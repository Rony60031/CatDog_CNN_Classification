import os
import time
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from model import Model_CatDog
from prepare_data import CatDogDataset, transform

# 1. Device
device = torch.device("cuda")

# 2. Dataset
train_dir = r"D:\data\catdog\training_set"

train_dataset = CatDogDataset(
    train_dir, transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# 3. Model
model = Model_CatDog().to(device)

# 4. Loss
criterion = nn.CrossEntropyLoss()

# 5. Adam Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# 6. Training
EPOCHS = 5

print("Optimizer: Adam")
print("Learning Rate: 0.0001")
print("Device:", device)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    start = time.time()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    torch.cuda.synchronize()

    epoch_loss = total_loss / total
    accuracy = 100 * correct / total

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("Loss:", round(epoch_loss, 4))
    print("Accuracy:", round(accuracy, 2), "%")
    print("Time:", round((time.time()-start)/60, 2), "minutes")

    os.makedirs("model", exist_ok=True)

    torch.save(
        model.state_dict(),
        f"model/adam_epoch_{epoch+1}.pth"
    )

print("\nAdam Training Test Completed!")