import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # kernel
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=4, stride=2, padding=2)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, stride=2)

        # batch normalization
        # normalize conv1, conv3, conv5, conv7, conv4 8 output channels
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv3_bn = nn.BatchNorm2d(8)
        self.conv4_bn = nn.BatchNorm2d(8)

        #drop out layers for conv4
        self.conv4_dol = nn.Dropout(p=0.1)

        # 2 fully connected layer 
        self.fc1 = nn.Linear(288, 100)
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def num_flat_features(self, x):
        size = x[0].size() # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # apply batch normalization after applying 1st conv
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.conv4_dol(self.conv4_bn(F.relu(self.conv4(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)
