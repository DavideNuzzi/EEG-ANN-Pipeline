import torch.nn as nn
from models.base_models import BaseClassifier


class EEGNetAdaptive(BaseClassifier):

    def __init__(self, num_classes, channels, num_timepoints, fs, features, depth=1, metrics=None):
        super(EEGNetAdaptive, self).__init__(metrics)

        self.num_classes = num_classes
        self.channels = channels

        kernel_size = fs // 2
        if kernel_size > num_timepoints // 2:
            kernel_size = num_timepoints // 2

        starting_block = [
            nn.Conv2d(1, features, kernel_size=(1, kernel_size), padding='same', bias=False), 
            nn.BatchNorm2d(features),
            nn.Conv2d(features, features, kernel_size=(channels, 1), bias=False, groups=features),
            nn.BatchNorm2d(features),
            nn.ELU(), 
            nn.Dropout(0.5),
            nn.Flatten(start_dim=2, end_dim=-1)
        ]

        middle_block = []

        for i in range(depth):
            kernel_size = kernel_size // 2
            if kernel_size > 1:
                middle_block.extend([
                    nn.Conv1d(features, features, kernel_size=kernel_size, bias=False),
                    nn.BatchNorm1d(features),
                    nn.ELU(),
                    nn.Conv1d(features, features, kernel_size=1, bias=False),
                    nn.BatchNorm1d(features),
                    nn.ELU(),
                    nn.Dropout(0.5)
                ])
            else:
                break

        last_block = [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(features, num_classes)
        ]
        self.starting_layers = nn.Sequential(*starting_block)
        self.layers = nn.Sequential(*starting_block, *middle_block, *last_block)
        
    def forward(self, x):
        x = self.layers(x)
        return x
