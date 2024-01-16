import torch
import torch.nn as nn
import sys
from shapley.models.resnet import BasicBlock, conv3x3, FitModule
import torch.nn.functional as F
import numpy as np

class ResNet_Extractor(FitModule):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Extractor, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(11,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def score(self, X, y):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        y_pred = self.predict(X)
        return np.mean(y.numpy() == np.argmax(y_pred.numpy(), axis=1))


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out_deep = out.view(out.size(0), -1)
        return out_deep

class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(input_dim, input_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(input_dim * 2, output_dim)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EstimatorNetwork(nn.Module):
    def __init__(self, d_model = 512, dropout = 0.1, num_classes=10):
        super(EstimatorNetwork, self).__init__()

        # assert query_dim == value_dim, "Query and value dimension should be same"

        self.resnet = ResNet_Extractor(BasicBlock, [2,2,2,2], num_classes)
        self.fc1 = FeedForward(d_model,256,dropout)
        self.fc2 = FeedForward(256, 128, dropout)
        self.fc3 = FeedForward(128, 64, dropout)
        self.fc4   = FeedForward(64,1, dropout)
        

    def forward(self, input):
        result = self.resnet(input)
        result = self.fc1(result)
        result = self.fc2(result)
        result = self.fc3(result)
        result = self.fc4(result)
        return result