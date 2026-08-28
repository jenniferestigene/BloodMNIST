
"""
model.py

Defines the CNN architecture used to classify BloodMNIST images into 
8 blood cell types.
"""



import torch
import torch.nn as nn

class BloodCellCNN(nn.Module):
    """
    3-layer CNN for 8-class blood cell classification on 64x64 
    RGB microscopy images (BloodMNIST).

    Uses kernel_size=3 with padding=1 in each conv layer.
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()


        # Block 1: 28x28 -> conv (28x28) -> pool(14x14)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 14x14 -> conv (14x14) -> pool (7x7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 7x7 -> conv (7x7) -> pool (3x3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Shared pooling layer - the same operation reused after each conv
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout before the classifier head
        self.dropout = nn.Dropout(p=0.3)

        # Flattened dimension: 128 channels x 3 x 3 spatial = 1,152
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)


    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Flatten (N, 128, 3, 3) -> (N, 1152)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = BloodCellCNN(num_classes=8)
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")