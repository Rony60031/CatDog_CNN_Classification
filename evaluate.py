import os
import torch
from torch.utils.data import DataLoader

from model import Model_CatDog
from prepare_data import CatDogDataset, transform

# 1. Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# 2. Test Dataset
test_dir = r"D:\data\catdog\test_set"

test_dataset = CatDogDataset(
    test_dir,
    transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

# 3. Load trained model
model = Model_CatDog().to(device)

checkpoint_path = "model/adam_epoch_5.pth"

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()

# 4. Evaluation
correct = 0
total = 0

# Rows = Actual class, Columns = Predicted class
confusion = torch.zeros(2, 2, dtype=torch.int64)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

        indices = labels * 2 + predictions

        confusion += torch.bincount(
            indices.cpu(),
            minlength=4
        ).reshape(2, 2)

# 5. Final accuracy
accuracy = 100 * correct / total

print("\n========== TEST RESULTS ==========")

print("Total test images:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)

print("Test Accuracy: {:.2f}%".format(accuracy))

print("\nConfusion Matrix:")
print(confusion)

print("\nCat correct:", confusion[0, 0].item())
print("Cat incorrect:", confusion[0, 1].item())

print("Dog correct:", confusion[1, 1].item())
print("Dog incorrect:", confusion[1, 0].item())

print("\nTesting Completed!")