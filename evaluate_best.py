import torch
from torch.utils.data import DataLoader

from model import Model_CatDog
from prepare_data import CatDogDataset, transform

# Device
device = torch.device("cuda")

# Test dataset
test_dir = r"D:\data\catdog\test_set"

test_dataset = CatDogDataset(test_dir, transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

# Load best model
model = Model_CatDog().to(device)

model.load_state_dict(
    torch.load(
        "model/best_catdog_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

correct = 0
total = 0

confusion = torch.zeros(2, 2, dtype=torch.int64)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        indices = labels * 2 + predictions

        confusion += torch.bincount(
            indices.cpu(),
            minlength=4
        ).reshape(2, 2)

# Final metrics
accuracy = 100 * correct / total

cat_precision = (
    confusion[0, 0].item() /
    max(confusion[:, 0].sum().item(), 1)
)

cat_recall = (
    confusion[0, 0].item() /
    max(confusion[0, :].sum().item(), 1)
)

dog_precision = (
    confusion[1, 1].item() /
    max(confusion[:, 1].sum().item(), 1)
)

dog_recall = (
    confusion[1, 1].item() /
    max(confusion[1, :].sum().item(), 1)
)

print("\n========== BEST MODEL TEST RESULTS ==========")

print("Best checkpoint: Epoch 6")
print("Total images:", total)
print("Correct:", correct)
print("Incorrect:", total - correct)

print("Final Test Accuracy: {:.2f}%".format(accuracy))

print("\nConfusion Matrix:")
print(confusion)

print("\nCat Precision: {:.2f}%".format(cat_precision * 100))
print("Cat Recall: {:.2f}%".format(cat_recall * 100))

print("Dog Precision: {:.2f}%".format(dog_precision * 100))
print("Dog Recall: {:.2f}%".format(dog_recall * 100))

print("\nFinal Evaluation Completed!")