"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
import os, hashlib
import lpips
import requests
import tqdm
from src.utils.transforms import *

def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()
def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm.tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)
def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path
URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}



class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "/workspace/lxc/vqgan/taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)


import torch
import torch.nn as nn
import torch.nn.functional as F



class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight
# def calculate_adaptive_weight( nll_loss, g_loss, last_layer=None):
#     nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
#     g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]


#     d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
#     d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
#     # d_weight = d_weight * self.discriminator_weight
#     d_weight = d_weight * 1.0
#     return d_weight

def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None):
    # 计算 nll_loss 和 g_loss 相对于 last_layer 的梯度
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True, allow_unused=True)
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True, allow_unused=True)

    # 检查是否有梯度
    if nll_grads[0] is None or g_grads[0] is None:
        # 如果没有梯度，返回一个默认的 d_weight
        # print("Warning: No gradients for nll_loss or g_loss, returning default weight.")
        d_weight = torch.tensor(1.0).to(nll_loss.device)  # 默认的适应性权重
    else:
        # 如果有梯度，计算 d_weight
        nll_grads = nll_grads[0]
        g_grads = g_grads[0]
        
        # 计算基于梯度的适应性权重
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        
        # 对 d_weight 进行限制，避免其值过大
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * 1.0  # 可以根据需要调整系数
        
    return d_weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


import random

class ALLLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        # self.gan = GANLoss()
        # self.style = StyleLoss()
        # self.lpips =  ps.PerceptualLoss(model='net-lin', net='vgg',use_gpu=torch.cuda.is_available(),gpu_ids=gpu_id)
        # self.lpips =  ps.PerceptualLoss(model='net-lin', net='alex',use_gpu=torch.cuda.is_available(),gpu_ids=gpu_id)
        self.lpips = lpips.LPIPS(net='vgg').cuda()

        
        self.lmbda = lmbda
        self.l1 = nn.L1Loss(reduction='none')
        # self.vgg = Vgg16().to(device).eval()
        self.mse = nn.MSELoss(reduction='none')

    
    
    def padding_and_trimming(self,
        x_rec, # decoder output
        x # reference image
        ):
        _, _, H, W = x.size()
        x_rec = F.pad(x_rec, (15, 15, 15, 15), mode='replicate')
        x = F.pad(x, (15, 15, 15, 15), mode='replicate')
        _, _, H_pad, W_pad = x.size()
        top = random.randrange(0, 16)
        bottom = H_pad - random.randrange(0, 16)
        left = random.randrange(0, 16)
        right = W_pad - random.randrange(0, 16)
        x_rec = F.interpolate(x_rec[:, :, top:bottom, left:right],
        size=(H, W), mode='bicubic', align_corners=False)
        x = F.interpolate(x[:, :, top:bottom, left:right],
        size=(H, W), mode='bicubic', align_corners=False)
        return x_rec, x

    def forward(self, rec, target):
        N, _, H, W = target.size()
        out = {}
        out["l1_loss"]  = torch.abs( rec.contiguous() -  target.contiguous()).mean(dim=[1, 2, 3])

        x_pad, target_pad = self.padding_and_trimming(rec, target)
        out["lpips"] = self.lpips(x_pad, target_pad, normalize=True) # input is 0 ~ 1
        out["lpips"] = torch.mean(out["lpips"], (1, 2, 3))
        #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        out["mse_loss"] = self.mse(rec, target)
        out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        
        # print("l1_loss", out["l1_loss"].mean())
        # print("lpips", out["lpips"].mean())
        return out
    
class ALLYUVLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        # self.gan = GANLoss()
        # self.style = StyleLoss()
        # self.lpips =  ps.PerceptualLoss(model='net-lin', net='vgg',use_gpu=torch.cuda.is_available(),gpu_ids=gpu_id)
        # self.lpips =  ps.PerceptualLoss(model='net-lin', net='alex',use_gpu=torch.cuda.is_available(),gpu_ids=gpu_id)
        self.lpips = lpips.LPIPS(net='vgg').cuda()

        
        self.lmbda = lmbda
        self.l1 = nn.L1Loss(reduction='none')
        # self.vgg = Vgg16().to(device).eval()
        self.mse = nn.MSELoss(reduction='none')

    
    
    def padding_and_trimming(self,
        x_rec, # decoder output
        x # reference image
        ):
        _, _, H, W = x.size()
        x_rec = F.pad(x_rec, (15, 15, 15, 15), mode='replicate')
        x = F.pad(x, (15, 15, 15, 15), mode='replicate')
        _, _, H_pad, W_pad = x.size()
        top = random.randrange(0, 16)
        bottom = H_pad - random.randrange(0, 16)
        left = random.randrange(0, 16)
        right = W_pad - random.randrange(0, 16)
        x_rec = F.interpolate(x_rec[:, :, top:bottom, left:right],
        size=(H, W), mode='bicubic', align_corners=False)
        x = F.interpolate(x[:, :, top:bottom, left:right],
        size=(H, W), mode='bicubic', align_corners=False)
        return x_rec, x

    def forward(self, rec, target):
        N, _, H, W = target.size()
        out = {}
        rec_rgb = ycbcr2rgb(rec)
        target_rgb = ycbcr2rgb(target)

        l1_rgb = torch.abs( rec_rgb.contiguous() -  target_rgb.contiguous()).mean(dim=[1, 2, 3])
        ll_yuv = torch.abs( rec.contiguous() -  target.contiguous()).mean(dim=[1, 2, 3])
        out["l1_loss"]  = 0.8 * ll_yuv + 0.2 * l1_rgb

        # x_pad, target_pad = self.padding_and_trimming(rec, target)
        out["lpips"] = self.lpips(rec_rgb, target_rgb, normalize=True) # input is 0 ~ 1
        out["lpips"] = torch.mean(out["lpips"], (1, 2, 3))
        #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        
        return out