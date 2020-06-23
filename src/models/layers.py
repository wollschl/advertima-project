import torch.nn as nn
from torch.nn.functional import relu

class ConvLayer (nn.Module):
    def __init__(self, c_in, c_out, stride, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(c_in,c_out,
                              stride=stride,
                              kernel_size=kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return relu(self.bn(self.conv(x)))