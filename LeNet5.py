import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # quote: "Layer C1 is a convolutional layer with 6 feature maps.
        # Each unit in each feature map is connected to a 5x5 neighborhood in the input
        # (page: 7 , title: LeNet-5, 3. paragraph)"
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # quote: Layer S2 is a sub-sampling layer with 6 feature maps of
        # size 14x14. Each unit in each feature map is connected to a
        # 2x2 neighborhood in the corresponding feature map in C1.   (page: 7, title: LeNet-5, 4. paragraph)"
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

        # quote: "Layer C3 is a convolutional layer with 16 feature maps.
        # Each unit in each feature map is connected to several 5x5 neighborhoods at identical locations in a subset of S2's feature maps.
        # (page: 7 , title: LeNet-5, 5. paragraph)""
        # (i added padding for shape mismatch)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=1)


        # output features of fully connected layers can be seen in fig. 2 (page: 7)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)

        self.fc2 = nn.Linear(in_features=120, out_features=84)

        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)  # Use 95 if modeling full ASCII

    def forward(self, x):
        #convolution
        #i changed activation function to gelu because it performed better
        x = F.gelu(self.conv1(x))
        x = self.avg_pool(x)
        x = F.gelu(self.conv2(x))
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)  # flatten

        #fully connected layers
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x
