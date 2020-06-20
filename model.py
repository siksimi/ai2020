import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pdb import set_trace as st

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nch = 32
        nch = self.nch
        self.conv1 = nn.Conv2d(1, nch, 5, stride = 2)
        self.bn1 = nn.BatchNorm2d(nch)
        self.conv2 = nn.Conv2d(nch, nch * 2, 3)
        self.bn2 = nn.BatchNorm2d(nch * 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(nch * 2 , nch * 4, 3)
        self.bn3 = nn.BatchNorm2d(nch * 4)
        self.conv4 = nn.Conv2d(nch * 4 , nch * 4, 3)
        self.bn4 = nn.BatchNorm2d(nch * 4)
        self.conv5 = nn.Conv2d(nch * 4 , nch * 8, 3)
        self.bn5 = nn.BatchNorm2d(nch * 8)
        self.conv6 = nn.Conv2d(nch * 8 , nch * 8, 3)
        self.bn6 = nn.BatchNorm2d(nch * 8)
        self.conv7 = nn.Conv2d(nch * 8 , nch * 16, 3)
        self.bn7 = nn.BatchNorm2d(nch * 16)
        self.conv8 = nn.Conv2d(nch * 16 , nch * 16, 3)
        self.adap = nn.AdaptiveAvgPool2d (1)
        self.bn8 = nn.BatchNorm2d(nch * 16)
        self.fc1 = nn.Linear(nch * 16, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool(F.relu(self.bn8(self.conv8(x))))
        x = self.adap(x)
        x = x.view(-1, self.nch * 16)
        # st()
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x