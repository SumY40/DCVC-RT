from turtle import forward
import torch
import torch.nn as nn
from .layers import MetaNeXtStage, CheckerboardMaskedConv2d
from .base import CompressionModel
import torch.nn.functional as F
from .utils import conv, deconv, quantize_ste, Demultiplexer, Multiplexer, CharbonnierLoss, VGGLoss_ESRGAN
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

class ScalingNet(nn.Module):
    def __init__(self, channel):
        super(ScalingNet, self).__init__()
        self.channel = int(channel)
        self.fc1 = nn.Linear(1, channel // 2, bias=True)
        self.fc2 = nn.Linear(channel // 2, channel, bias=True)
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x, lambda_rd):
        b, c, _, _ = x.size()
        scaling_vector = torch.exp(10 * self.fc2(F.relu(self.fc1(lambda_rd))))
        scaling_vector = scaling_vector.view(b, c, 1, 1)
        x_scaled = x * scaling_vector.expand_as(x)
        return x_scaled
    
class ORP_decoder(nn.Module):


    def __init__(self, N=128, M=320, w_adv=0.05, w_vgg=1, model_size = "80M"):
        super().__init__()

        self.num_class = 20
        depths = [2, 2, 6, 2, 2, 2]
        self.num_slices = 5
        self.w_adv = w_adv
        self.w_vgg = w_vgg
        self.model_size = model_size

 
        N = 96
        M = 128
        self.M = M
        in_ch_list = [0, 8, 16, 32, 64]
        out_ch_list = [8, 8, 16, 32, self.M - 64]

        self.g_s_scale0 = ScalingNet(channel=M)
        self.g_s_mr0 = ScalingNet(channel=M)
        self.g_s0 = MetaNeXtStage(dim=M, depth=depths[2])
        self.g_s1 = deconv(M, N * 2, kernel_size=3, stride=2)

        self.g_s_scale1 = ScalingNet(channel=N * 2)
        self.g_s_mr1 = ScalingNet(channel=N * 2)
        self.g_s2 = MetaNeXtStage(dim=N * 2, depth=depths[3])
        self.g_s3 = deconv(N * 2, N * 3 // 2, kernel_size=3, stride=2)

        self.g_s_scale2 = ScalingNet(channel=N * 3 // 2)
        self.g_s_mr2 = ScalingNet(channel=N * 3 // 2)
        self.g_s4 = MetaNeXtStage(dim=N * 3 // 2, depth=depths[4])
        self.g_s5 = deconv(N * 3 // 2, N, kernel_size=3, stride=2)

        self.g_s_scale3 = ScalingNet(channel=N)
        self.g_s_mr3 = ScalingNet(channel=N)
        self.g_s6 = MetaNeXtStage(dim=N, depth=depths[5])
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        

    def g_s(self, x, lambda_rd):
        x = self.g_s_scale0(x, lambda_rd)
        x = self.g_s0(x)
        x = self.g_s1(x)
        x = self.g_s_scale1(x, lambda_rd)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s_scale2(x, lambda_rd)
        x = self.g_s4(x)
        x = self.g_s5(x)
        x = self.g_s_scale3(x, lambda_rd)
        x = self.g_s6(x)
        x = self.g_s7(x)
        return x


    def forward(self, x, lambda_rd):

        r = self.g_s(x, lambda_rd)

        return r