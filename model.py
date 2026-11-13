import torch
import torch.nn as nn


class Model_CatDog(nn.Module):

    def __init__(self):
        super().__init__()

        # Four Convolution Layers
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=5, padding=2
        )

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1
        )

        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1
        )

        self.conv4 = nn.Conv2d(
            256, 256, kernel_size=5
        )

        # Three Fully Connected Layers
        self.fc1 = nn.Linear(256 * 12 * 12, 2048)

        self.fc2 = nn.Linear(2048, 512)

        self.fc3 = nn.Linear(512, 2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # Convolution + ReLU + Max Pooling
        x = nn.functional.max_pool2d(
            nn.functional.relu(self.conv1(x)), 2
        )

        x = nn.functional.max_pool2d(
            nn.functional.relu(self.conv2(x)), 2
        )

        x = nn.functional.max_pool2d(
            nn.functional.relu(self.conv3(x)), 2
        )

        x = nn.functional.max_pool2d(
            nn.functional.relu(self.conv4(x)), 2
        )

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully Connected Layers
        x = self.fc1(x)
        x = nn.functional.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = nn.functional.relu(x)

        x = self.fc3(x)

        return x


# Test the CNN Model
if __name__ == "__main__":

    model = Model_CatDog()

    print("CNN Model Created Successfully!")

    # Count parameters
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("Total Parameters:", total_params)

    # Test input
    sample = torch.randn(2, 3, 224, 224)

    model.eval()

    with torch.no_grad():
        output = model(sample)

    print("Input Shape:", sample.shape)
    print("Output Shape:", output.shape)

    print("Model Architecture Test Successful!")