import torch.nn as nn
from models.base_models import BaseClassifier
from models.layers import Conv2dMaxNorm


class DeepNet(BaseClassifier):

    def __init__(self, num_classes, channels):
        super(DeepNet, self).__init__()

        self.num_classes = num_classes
        self.channels = channels

        conv_filters=(25, 50, 100, 200)

        blocks = []

        blocks.append(Conv2dMaxNorm(1, conv_filters[0], kernel_size=(1,5), max_norm_val=2))

        for i, filters in enumerate(conv_filters):
            if i == 0:
                kernel_size = (channels, 1)
                ch = 25
            else:
                kernel_size = (1,10)
                ch = conv_filters[i-1] 

            conv_block = nn.Sequential(
                Conv2dMaxNorm(ch, filters, kernel_size=kernel_size, max_norm_val=2),
                nn.BatchNorm2d(filters, eps=1e-5, momentum=0.9),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)),
                nn.Dropout(0.5)
            )
            blocks.append(conv_block)

        self.main_layers = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, x):

        x = self.main_layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
