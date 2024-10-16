import torch
import torch.nn as nn

class Conv2dMaxNorm(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, max_norm_val=3.0):
        
        super(Conv2dMaxNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.max_norm_val = max_norm_val

    def forward(self, x):

        # Apply the convolution
        out = self.conv(x)
        
        # Enforce max_norm constraint on the weights
        with torch.no_grad():
            self.conv.weight.data = torch.renorm(self.conv.weight, p=2, dim=0, maxnorm=self.max_norm_val)
        
        return out
        

# Come il blocco di EEGNet, ma uso N filtri per ogni canale separatamente
class EEGConvBlock(nn.Module):

    def __init__(self, channels, F1, F2, kernel_time):

        super(EEGConvBlock, self).__init__()

        self.conv_depthwise = nn.Conv1d(channels, channels * F1, kernel_size = kernel_time, padding='same', groups=channels, bias=False)
        self.conv = nn.Conv2d(F1, F2, kernel_size = (channels, 1), bias=False)
        self.conv_pointwise = nn.Conv2d(F2, F2, kernel_size = (1,1), bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(channels * F1)
        self.batch_norm_2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()

    def forward(self, x):

        batch_size, _, num_channels, num_timepoints = x.shape
        
        x = x.view(batch_size, num_channels, num_timepoints)
        x = self.conv_depthwise(x)
        x = self.batch_norm_1(x)
        x = x.view((batch_size, num_channels, -1, num_timepoints))
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.conv(x) # (batch_size, filters*2, 1, N_T)
        x = self.conv_pointwise(x) # (batch_size, filters*2, 1, N_T)
        x = self.batch_norm_2(x) 
        x = self.elu(x)
        x = x.view(batch_size, 1, -1, num_timepoints)
        
        return x
        



# Come il precedente, ma lavoro con filtri di dimensione diversa
class EEGConvBlockMultiscale(nn.Module):

    def __init__(self, channels, F1, F2, kernel_time):

        super(EEGConvBlockMultiscale, self).__init__()

        self.conv_depthwise_1 = nn.Conv1d(channels, channels * F1, kernel_size = kernel_time, padding='same', groups=channels, bias=False)
        self.conv_depthwise_2 = nn.Conv1d(channels, channels * F1, kernel_size = kernel_time//2, padding='same', groups=channels, bias=False)
        self.conv_depthwise_3 = nn.Conv1d(channels, channels * F1, kernel_size = kernel_time//4, padding='same', groups=channels, bias=False)

        self.conv = nn.Conv2d(F1 * 3, F2, kernel_size = (channels, 1), bias=False)
        self.conv_pointwise = nn.Conv2d(F2, F2, kernel_size = (1,1), bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(channels * F1)
        self.batch_norm_2 = nn.BatchNorm1d(channels * F1)
        self.batch_norm_3 = nn.BatchNorm1d(channels * F1)

        self.batch_norm_4 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()

    def forward(self, x):

        batch_size, _, num_channels, num_timepoints = x.shape
        
        x = x.view(batch_size, num_channels, num_timepoints)

        x1 = self.conv_depthwise_1(x)
        x1 = self.batch_norm_1(x1)
        x1 = x1.view((batch_size, num_channels, -1, num_timepoints))

        x2 = self.conv_depthwise_2(x)
        x2 = self.batch_norm_2(x2)
        x2 = x2.view((batch_size, num_channels, -1, num_timepoints))

        x3 = self.conv_depthwise_3(x)
        x3 = self.batch_norm_3(x3)
        x3 = x3.view((batch_size, num_channels, -1, num_timepoints))

        x = torch.cat((x1,x2,x3), axis=2)
        x = torch.permute(x, (0, 2, 1, 3))

        x = self.conv(x) # (batch_size, filters*2, 1, N_T)
        x = self.conv_pointwise(x) # (batch_size, filters*2, 1, N_T)
        x = self.batch_norm_4(x) 
        x = self.elu(x)
        x = x.view(batch_size, 1, -1, num_timepoints)
        
        return x


