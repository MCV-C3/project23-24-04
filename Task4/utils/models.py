import torch
import torch.nn as nn


class BlockedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        out_channels = 16
        dropout_rate = 0.5
        hidden_dim = 128

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fourth convolutional block
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Fully connected layer
        self.fc = nn.Linear(15376, hidden_dim * 4)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # Assuming x has the shape (batch_size, in_channels, height, width)
        # First convolutional block
        x1 = self.conv_block1(x)

        # Second convolutional block with skip connection
        x2 = self.conv_block2(x1)

        x2 += x1  # Skip connection
        x2 = nn.functional.relu(x2)

        # Third convolutional block with skip connection
        x3 = self.conv_block3(x2)
        x3 += x2  # Skip connection
        x3 = nn.functional.relu(x3)

        x3 = self.gap(x1)
        # Flatten the tensor before feeding it into the fully connected layer
        x3 = x1.view(x3.size(0), -1)

        # Fully connected layer
        x4 = self.fc(x3)

        return x4


class LightweightCNN(nn.Module):
    def __init__(self, input_channels=3, output_features=128):
        super(LightweightCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, output_features), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class StrideCNN(nn.Module):
    def __init__(self, input_channels=3, output_features=128):
        super(StrideCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=4, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, output_features), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


class StrideCNN_v2(nn.Module):
    def __init__(self, input_channels=3, output_features=128):
        super(StrideCNN_v2, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=1, stride=2, padding=0)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(16 * 8 * 8, output_features)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class DeepCNN(nn.Module):
    def __init__(self, input_channels=3, output_features=128):
        super(DeepCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(16 * 4 * 4, output_features)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class ShallowCNN(nn.Module):
    def __init__(self, input_channels=3, output_features=128):
        super(ShallowCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=4, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(32 * 16 * 16, output_features)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    from torchvision import transforms as transforms
    import numpy as np
    from torchsummary import summary
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision

    convnet = DeepCNN()
    convnet.cuda()
    print(summary(convnet, (3, 256, 256)))
