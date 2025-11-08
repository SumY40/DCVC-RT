import functools
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import init

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        input = input.contiguous() # 确保输入连续
        return self.main(input)


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h


class GANLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, D_real_logits, D_gen_logits,mode='generator_loss'):
        D_loss_real = F.binary_cross_entropy_with_logits(input=D_real_logits,target=torch.ones_like(D_real_logits),reduction='none')
        D_loss_gen = F.binary_cross_entropy_with_logits(input=D_gen_logits,target=torch.zeros_like(D_gen_logits),reduction='none')
        D_loss = D_loss_real + D_loss_gen
        G_loss = F.binary_cross_entropy_with_logits(input=D_gen_logits,target=torch.ones_like(D_gen_logits),reduction='none')
        loss = G_loss if mode == 'generator_loss' else D_loss / 2.0
        return loss


class DiscriminatorHIFICnomean(nn.Module):
    # 40M大小:
    # def __init__(self, image_dims=(3,256,256), context_dims=(128,16,16), C=64, spectral_norm=True):
    # 80M大小:
    def __init__(self, image_dims=(3,256,256), context_dims=(128,16,16), C=128, spectral_norm=True):
        """ 
        Convolutional patchGAN discriminator used in [1].
        Accepts as input generator output G(z) or x ~ p*(x) where
        p*(x) is the true data distribution.
        Contextual information provided is encoder output y = E(x)
        ========
        Arguments:
        image_dims:     Dimensions of input image, (C_in,H,W)
        context_dims:   Dimensions of contextual information, (C_in', H', W')
        C:              Bottleneck depth, controls bits-per-pixel
                        C = 220 used in [1], C = C_in' if encoder output used
                        as context.

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        super(DiscriminatorHIFICnomean, self).__init__()
        
        self.image_dims = image_dims
        self.context_dims = context_dims
        im_channels = self.image_dims[0]
        kernel_dim = 4
        context_C_out = 12
        filters = (64, 128, 256, 512)

        # Upscale encoder output - (C, 16, 16) -> (12, 256, 256)
        self.context_conv = nn.Conv2d(C, context_C_out, kernel_size=3, padding=1, padding_mode='reflect')
        self.context_upsample = nn.Upsample(scale_factor=16, mode='nearest')

        # # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        # # assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # # Layer / normalization options
        # # TODO: calculate padding properly
        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        # (C_in + C_in', 256,256) -> (64,128,128), with implicit padding
        # TODO: Check if removing spectral norm in first layer works
        self.conv1 = norm(nn.Conv2d(im_channels * 2 + context_C_out, filters[0], kernel_dim, **cnn_kwargs))
        # self.conv1 = norm(nn.Conv2d(im_channels + im_channels , filters[0], kernel_dim, **cnn_kwargs))

        # (128,128) -> (64,64)
        self.conv2 = norm(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))

        # (64,64) -> (32,32)
        self.conv3 = norm(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))

        # (32,32) -> (16,16)
        self.conv4 = norm(nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))

        self.conv_out = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1)

        #  OASIS dis
        # self.opt = opt
        # #sp_norm = nn.utils.weight_norm
        # self.output_channel = opt.semantic_nc + 1 # for N+1 loss
        # self.channels = [3, 128, 128, 256, 256, 512, 512]
        # self.body_up   = nn.ModuleList([])
        # self.body_down = nn.ModuleList([])
        # self.first_con = nn.Conv2d(24, out_channels=3, kernel_size=1)
        # for i in range(opt.num_res_blocks):
        #     self.body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
        # # decoder part
        # self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2], opt, 1))
        # for i in range(1, opt.num_res_blocks-1):
        #     self.body_up.append(residual_block_D(2*self.channels[-1-i], self.channels[-2-i], opt, 1))
        # self.body_up.append(residual_block_D(2*self.channels[1], 64, opt, 1))
        # self.layer_up_last = nn.Conv2d(64, self.output_channel, 1, 1, 0)

        # self.project_layer = nn.Conv2d(64, 1, kernel_size=1)

        # self.last_con = nn.Conv2d(64, 128, 1)

    def forward(self, x, y):
        """
        x: Concatenated real/gen images
        y: Quantized latents
        """
        batch_size = x.size()[0]
        x_h, x_w = x.size(2), x.size(3)
        y_h, y_w = y.size(2), y.size(3)
        scale_factor_h, scale_factor_w = x_h // y_h, x_w // y_w
        # Concatenate upscaled encoder output y as contextual information
        y = self.activation(self.context_conv(y))
        y = nn.functional.interpolate(y, scale_factor=(scale_factor_h, scale_factor_w), mode='nearest')
        x = torch.cat((x,y), dim=1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        
        # out_logits = self.conv_out(x).view(batch_size,-1,1,1)
        out_logits = self.conv_out(x).view(batch_size,-1)


        return out_logits
    
    
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
from torch.nn.utils import spectral_norm

class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, C=128, use_sn=True):
        super(PatchGANDiscriminator, self).__init__()
        filters = (64, 128, 256, 512)
        kernel_dim = 4
        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        # 👇改这里
        norm_fn = nn.utils.spectral_norm if use_sn else nn.utils.weight_norm

        self.conv1 = norm_fn(nn.Conv2d(in_channels, filters[0], kernel_dim, **cnn_kwargs))
        self.conv2 = norm_fn(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))
        self.conv3 = norm_fn(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))
        self.conv4 = norm_fn(nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))
        self.conv_out = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        return self.conv_out(x)  # [B, 1, H, W]

class DINOv2Backbone(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        self.backbone  = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.embed_dim = self.backbone.embed_dim  # e.g. 1024 for vitl14

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = resize_to_multiple_of_patch(x, patch_size=14)  # 自定义函数

        features = self.backbone.forward_features(x)       # 返回 dict
        reg_tokens = features['x_norm_regtokens']          # [B, N, D]
        pooled = reg_tokens.mean(dim=1)                    # [B, D]
        return pooled  # 你也可以返回 reg_tokens 直接送入更强判别器
def resize_to_multiple_of_patch(x, patch_size=14):
    B, C, H, W = x.shape
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size
    if new_H != H or new_W != W:
        x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
    return x


from torch.nn.utils import spectral_norm

class MLPD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        print(activation)
        self.decoder = nn.Sequential(spectral_norm(nn.Linear(in_ch, out_ch)),
                                     activation,
                                     )
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)

    def forward(self, x, c):
        h = self.decoder(x)
        out = self.out(h)
        if self.embed is not None:
            out += torch.sum(self.embed(c) * h, 1, keepdim=True)
        return out

class DINOv2WithMLPDiscriminator(nn.Module):
    def __init__(self, freeze_backbone=True, num_classes=0):
        super().__init__()
        self.backbone = DINOv2Backbone(freeze_backbone=freeze_backbone)
        self.head = MLPD(in_ch=self.backbone.embed_dim, out_ch=256, num_classes=num_classes)

    def forward(self, x, c=None):
        feat = self.backbone(x)      # (B, D)
        score = self.head(feat, c)   # (B, 1)
        return score



import open_clip

clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda')
clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda')

class CLIPBackbone(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained='openai', freeze_backbone=True):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = self.model.visual
        self.embed_dim = self.visual.output_dim

        if freeze_backbone:
            for p in self.visual.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: [B,3,H,W], 范围 [0,1]
        x = F.interpolate(x, size=(224,224), mode="bicubic", align_corners=False)
        x = (x - clip_mean[None,:,None,None]) / clip_std[None,:,None,None]  # normalize
        feat = self.model.encode_image(x)
        return feat
    
class CLIPWithMLPDiscriminator(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained="/workspace/lxc/DCVC/RT/checkpoints/open_clip_pytorch_model.bin", freeze_backbone=True, num_classes=0):
        super().__init__()
        self.backbone = CLIPBackbone(model_name, pretrained, freeze_backbone)
        self.head = MLPD(in_ch=self.backbone.embed_dim, out_ch=256, num_classes=num_classes)

    def forward(self, x, c=None):
        feat = self.backbone(x)      # [B, D]
        score = self.head(feat, c)   # [B, 1]
        return score
