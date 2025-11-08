import torch.nn as nn

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

class ResInv(nn.Module):
    def __init__(self, N=3, **kwargs):
        super().__init__()
        
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=1),
        )
        
        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            conv3x3(N, 3, 1),
        )
    
    def forward(self, x, rev=False):
        
        if not rev:
            out = self.g_a(x)
        
        else:
            out = self.g_s(x)

        return out