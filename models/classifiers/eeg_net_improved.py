import torch.nn as nn
from models.base_models import BaseClassifier
from models.layers import EEGConvBlock


# Voglio che ogni canale possa essere filtrato in maniera separata
# Quindi se faccio un reshape da (1, N_c, N_T) a (N_C, N_T)
# e poi applico una convoluzione 1D con N filtri, otterrò ()
# Voglio che, per ogni canale, si ottengano N versioni filtrate del segnale
# Il risultato deve essere un tensore (N, N_c, N_T)
# Questa sarà la mia convoluzione sui tempi, come la costruisco?
# Una volta fatto il reshape vorrei mettere dei gruppi
class EEGNetImproved(BaseClassifier):

    def __init__(self, num_classes, channels, fs):
        super(EEGNetImproved, self).__init__()

        self.num_classes = num_classes
        self.channels = channels
        self.fs = fs

        # filters_per_channel = 4
        # kernel_time = fs//4
        F1 = 8
        F2 = 8
        kernel_time = 64

        self.block_1 = EEGConvBlock(channels, F1, F2, kernel_time)
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=(1,4))
        self.dropout_1 = nn.Dropout(0.5)
        self.block_2 = EEGConvBlock(F2, F2, F2, kernel_time//4)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=(1,8))
        self.dropout_2 = nn.Dropout(0.5)  
        self.conv_final = nn.Conv2d(1, 1, (1, 8), bias=False)  
        self.batch_norm = nn.BatchNorm2d(1)   
        self.dropout_3 = nn.Dropout(0.5)  
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, x):

        x = self.block_1(x) # (b, 1, 2F, T)
        x = self.avg_pool_1(x) # (b, 1, 2F, T/4)
        x = self.dropout_1(x)
        x = self.block_2(x) # (b, 1, 2F, T/4)
        x = self.avg_pool_2(x) # (b, 1, 2F, T/32)
        x = self.dropout_2(x)
        x = self.conv_final(x)
        x = self.batch_norm(x)
        x = self.dropout_3(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x
