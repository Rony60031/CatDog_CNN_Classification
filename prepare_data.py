import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. Dataset paths
BASE_PATH = r"D:\data\catdog"

TRAIN_DIR = os.path.join(BASE_PATH, "training_set")
TEST_DIR = os.path.join(BASE_PATH, "test_set")

# 2. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3. Custom Dataset
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
            raise ValueError(
                f"Unknown image label: {filename}"
            )

        return image, label


# 4. Create datasets
train_dataset = CatDogDataset(
    TRAIN_DIR, transform
)

test_dataset = CatDogDataset(
    TEST_DIR, transform
)

# 5. Create DataLoaders
batch_size = 20

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 6. Check dataset
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

# 7. Check one batch
images, labels = next(iter(train_loader))

print("\nImage Tensor Shape:", images.shape)
print("Label Tensor Shape:", labels.shape)

print("\nFirst 10 labels:", labels[:10].tolist())

print("\nPreprocessing completed successfully!")