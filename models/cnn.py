import torch
import torch.nn as nn


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=3, out_channels=16, kernel_size=1, stride=1)
#         self.conv2 = nn.Conv2d(
#             in_channels=16, out_channels=32, kernel_size=2, stride=1)
#         self.conv3 = nn.Conv2d(
#             in_channels=32, out_channels=64, kernel_size=1, stride=1)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 17 * 17, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         # print("After conv1", x.shape)
#         x = self.relu(self.conv2(x))
#         # print("After conv2", x.shape)
#         x = self.relu(self.conv3(x))
#         # print("After conv3", x.shape)
#         x = self.maxpool(x)
#         # print("After maxpool2d", x.shape)
#         x = self.flatten(x)
#         # print("After flatten", x.shape)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         # print("After fc1", x.shape)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         # print("After fc2", x.shape)
#         x = torch.sigmoid(self.fc3(x))
#         return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.batchnorm_fc1 = nn.BatchNorm1d(128)
        self.batchnorm_fc2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.relu(self.batchnorm_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


# Example instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
