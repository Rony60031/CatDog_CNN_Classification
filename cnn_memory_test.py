import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Model_CatDog
from prepare_data import CatDogDataset, transform

# 1. Select GPU
device = torch.device("cuda")

print("Using device:", torch.cuda.get_device_name(0))

# 2. Load dataset
train_dir = r"D:\data\catdog\training_set"

train_dataset = CatDogDataset(train_dir, transform)

# 3. Small batch size
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

# 4. Create model
model = Model_CatDog().to(device)

# 5. Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

# 6. Test one training batch
model.train()

print("\nStarting CNN training test...")

try:

    images, labels = next(iter(train_loader))

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    # Forward propagation
    outputs = model(images)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    torch.cuda.synchronize()

    print("\nTraining batch successful!")
    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)
    print("Loss:", loss.item())

    print(
        "GPU Memory Allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB"
    )

    print(
        "GPU Memory Reserved:",
        round(torch.cuda.memory_reserved() / 1024**3, 2),
        "GB"
    )

except torch.cuda.OutOfMemoryError:

    print("\nGPU MEMORY ERROR!")
    print("The model needs a smaller memory configuration.")

    raise

print("\nCNN Training Test Completed!")