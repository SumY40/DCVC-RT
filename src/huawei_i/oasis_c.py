import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class OASISResBlock(nn.Module):
    def __init__(self, fin, fout, up_or_down, first=False):
        super(OASISResBlock, self).__init__()
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        self.fmiddle = fout

        if self.first:
            self.conv1 = nn.Sequential(
                weight_norm(nn.Conv2d(fin, self.fmiddle, 3, padding=1)),
                nn.LeakyReLU(0.2)
            )
        else:
            layers = [nn.LeakyReLU(0.2)]
            if up_or_down > 0:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(weight_norm(nn.Conv2d(fin, self.fmiddle, 3, padding=1)))
            self.conv1 = nn.Sequential(*layers)

        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(self.fmiddle, fout, 3, padding=1))
        )

        if self.learned_shortcut:
            self.conv_s = weight_norm(nn.Conv2d(fin, fout, 1))
        else:
            self.conv_s = None

        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2, mode='nearest')
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = None

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0 and self.sampling is not None:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.sampling is not None:
            x = self.sampling(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        return x

class OASISDiscriminator(nn.Module):
    def __init__(self, input_dim_img=(256, 256, 3), input_dim_latent=(256 // 16, 256 // 16, 320), output_channel=135, num_res_blocks=6):
        super(OASISDiscriminator, self).__init__()
        h_x, w_x, c_x = input_dim_img
        h_y, w_y, c_y = input_dim_latent
        channels = [3, 128, 128, 256, 256, 512, 512]

        self.prepare_latent = nn.Sequential(
            weight_norm(nn.Conv2d(c_y, 64, 3, padding=1)),
            nn.LeakyReLU(0.2)
        )

        self.body_down = nn.ModuleList(
            [OASISResBlock(channels[i], channels[i + 1], -1, first=(i == 0)) for i in range(num_res_blocks)]
        )

        self.body_up = nn.ModuleList([
            OASISResBlock(channels[-1], channels[-2], 1),
            *[
                OASISResBlock(2 * channels[-i - 1], channels[-i - 2], 1) for i in range(1, num_res_blocks - 1)
            ],
            OASISResBlock(2 * channels[1], 64, 1)
        ])

        self.layer_up_last = nn.Conv2d(64, output_channel, kernel_size=1)

    def forward(self, x, y):
        # Prepare latent input
        latent = self.prepare_latent(y)
        latent = F.interpolate(latent, size=x.shape[2:], mode='nearest')

        # Encoder
        encoder_res = []
        for layer in self.body_down:
            x = layer(x)
            encoder_res.append(x)

        # Decoder
        x = self.body_up[0](x)
        for i, layer in enumerate(self.body_up[1:]):
            x = layer(torch.cat((encoder_res[-i - 1], x), dim=1))

        # Conditioning
        proj = torch.sum(latent * x, dim=1, keepdim=True)
        proj = proj.repeat(1, self.layer_up_last.out_channels, 1, 1)

        # Final output
        ans = self.layer_up_last(x) + proj
        return ans
