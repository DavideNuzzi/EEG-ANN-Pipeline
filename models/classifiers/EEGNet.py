import torch.nn as nn
from models.base_models import BaseClassifier
from models.layers import Conv2dMaxNorm


class EEGNet(BaseClassifier):

    def __init__(self, num_classes, channels):
        super(EEGNet, self).__init__()

        self.num_classes = num_classes
        self.channels = channels

        F1, F2 = 8, 16

        self.layers = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding='same', bias=False), # First temporal convolution
            nn.BatchNorm2d(F1),
            Conv2dMaxNorm(F1, F2, kernel_size=(channels, 1), bias=False, groups=F1, max_norm_val=1), # Depthwise convolution
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(0.5),
            nn.Conv2d(F2, F2, kernel_size=(1, 16), bias=False, groups=F2),  # Separable = Depthwise + Pointwise
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),              # Separable = Depthwise + Pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
