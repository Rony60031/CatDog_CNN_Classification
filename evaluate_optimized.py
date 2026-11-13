
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import Model_CatDog


# 1. Configuration
device = torch.device("cuda")

TEST_DIR = r"D:\data\catdog\test_set"

# 2. Test transformation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# 3. Dataset
class CatDogTestDataset(Dataset):

    def __init__(self, directory):
        self.directory = directory

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

        image = test_transform(image)

        if "cat" in filename.lower():
            label = 0
        elif "dog" in filename.lower():
            label = 1
        else:
            raise ValueError(filename)

        return image, label


# 4. Load test data
test_dataset = CatDogTestDataset(TEST_DIR)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

# 5. Load best optimized model
model = Model_CatDog().to(device)

model.load_state_dict(
    torch.load(
        "model/best_optimized_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# 6. Evaluation
confusion = torch.zeros(2, 2, dtype=torch.int64)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        indices = labels * 2 + predictions

        confusion += torch.bincount(
            indices.cpu(),
            minlength=4
        ).reshape(2, 2)


# 7. Metrics
total = confusion.sum().item()
correct = confusion.diag().sum().item()

accuracy = 100 * correct / total

print("\n========== OPTIMIZED CNN RESULTS ==========")

print("Total Test Images:", total)
print("Correct Predictions:", correct)
print("Incorrect Predictions:", total - correct)

print("Test Accuracy: {:.2f}%".format(accuracy))

print("\nConfusion Matrix:")
print(confusion)

for i, name in enumerate(["Cat", "Dog"]):

    tp = confusion[i, i].item()
    fp = confusion[:, i].sum().item() - tp
    fn = confusion[i, :].sum().item() - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-12)
    )

    print(f"\n{name} Metrics:")
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1-score: {:.2f}%".format(f1 * 100))

print("\nOptimized CNN Evaluation Completed!")
