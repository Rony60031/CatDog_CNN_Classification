import torch
import torch.nn as nn
import time

from torch.utils.data import DataLoader
from model import Model_CatDog
from prepare_data import CatDogDataset, transform

device = torch.device("cuda")

train_dir = r"D:\data\catdog\training_set"

dataset = CatDogDataset(train_dir, transform)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

model = Model_CatDog().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

model.train()

total_batches = 50
start_time = time.time()

print("Starting 50-batch speed test...")

for i, (images, labels) in enumerate(loader):

    if i >= total_batches:
        break

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    if (i + 1) % 10 == 0:
        print(
            f"Completed {i + 1}/{total_batches} batches"
        )

torch.cuda.synchronize()

elapsed = time.time() - start_time

print("\nSpeed Test Completed!")
print("Total batches:", total_batches)
print("Total time:", round(elapsed, 2), "seconds")
print("Average time per batch:", round(elapsed / total_batches, 2), "seconds")

estimated_epoch = (len(dataset) / 2) * (elapsed / total_batches)

print(
    "Estimated time for one epoch:",
    round(estimated_epoch / 60, 2),
    "minutes"
)

print(
    "GPU Memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)