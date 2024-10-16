import torch
import torch.nn as nn
from models.base_models import BaseClassifier
from models.layers import Conv2dMaxNorm


class ShallowNet(BaseClassifier):

    def __init__(self, num_classes, channels, filters=40):
        super(ShallowNet, self).__init__()

        self.num_classes = num_classes
        self.filters = filters
        self.channels = channels

        self.conv_1 = Conv2dMaxNorm(1, filters, kernel_size=(1, 13), max_norm_val=2)
        self.conv_2 = Conv2dMaxNorm(filters, filters, kernel_size=(self.channels, 1), bias=False, max_norm_val=2)
        self.batch_norm = nn.BatchNorm2d(filters)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 35), stride=(1, 7))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.batch_norm(x)
        x = torch.square(x)
        x = self.max_pool(x)
        x = torch.log(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
