import torch
import torch.nn as nn
from models.base_models import BaseClassifier
from models.layers import Conv2dMaxNorm


class TinyNet(BaseClassifier):

    def __init__(self, num_classes, channels, num_timepoints, dropout_rate, filters=32, metrics=None):
        super(TinyNet, self).__init__(metrics)

        self.conv_1 = nn.Conv2d(1, filters, kernel_size=(1, num_timepoints))
        self.batch_norm_1 = nn.BatchNorm2d(filters)
        self.conv_2 = nn.Conv2d(filters, filters, kernel_size=(channels, 1), groups=filters)
        self.batch_norm_2 = nn.BatchNorm2d(filters)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(filters, filters)
        self.selu = nn.SELU()
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(filters, num_classes)

    def forward(self, x):

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.selu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        return x
    
    