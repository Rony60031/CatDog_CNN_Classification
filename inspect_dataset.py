import os

# Main dataset path
DATASET_PATH = r"D:\data\catdog"

# Dataset folders
folders = [
    "training_set",
    "training_set2",
    "training_set3",
    "test_set",
    "test_set3"
]

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

print("=" * 55)
print("CAT AND DOG DATASET INSPECTION")
print("=" * 55)

for folder in folders:

    folder_path = os.path.join(DATASET_PATH, folder)

    print(f"\nFolder: {folder}")

    if not os.path.exists(folder_path):
        print("ERROR: Folder not found!")
        continue

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
        and os.path.isfile(os.path.join(folder_path, f))
    ]

    cats = [f for f in files if "cat" in f.lower()]
    dogs = [f for f in files if "dog" in f.lower()]

    unknown = len(files) - len(cats) - len(dogs)

    print("Total images:", len(files))
    print("Cat images:", len(cats))
    print("Dog images:", len(dogs))
    print("Unknown images:", unknown)

print("\n" + "=" * 55)
print("INSPECTION COMPLETED")
print("=" * 55)