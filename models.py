import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyModel).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True)
        self.conv2 =  = nn.Conv2d(in_channels, out_channels, 3, stride=1,
                 padding=0, dilation=1, groups=1, bias=False)

    def forward(self, *input):
        