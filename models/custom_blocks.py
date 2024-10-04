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