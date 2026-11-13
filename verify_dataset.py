import os
from PIL import Image

BASE_PATH = r"D:\data\catdog"

folders = ["training_set", "test_set"]

extensions = (".jpg", ".jpeg", ".png", ".bmp")

for folder in folders:

    folder_path = os.path.join(BASE_PATH, folder)

    print("\n" + "=" * 50)
    print("Checking:", folder)
    print("=" * 50)

    valid = 0
    invalid = 0
    sizes = {}

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(extensions):
            continue

        filepath = os.path.join(folder_path, filename)

        try:
            with Image.open(filepath) as img:
                img.verify()

            with Image.open(filepath) as img:
                size = img.size
                sizes[size] = sizes.get(size, 0) + 1

            valid += 1

        except Exception as e:
            print("Invalid image:", filename)
            invalid += 1

    print("Valid images:", valid)
    print("Invalid images:", invalid)
    print("Different image sizes:", len(sizes))

    print("\nMost common image sizes:")

    for size, count in sorted(
        sizes.items(),
        key=lambda item: item[1],
        reverse=True
    )[:5]:
        print(size, ":", count, "images")

print("\nDataset verification completed!")