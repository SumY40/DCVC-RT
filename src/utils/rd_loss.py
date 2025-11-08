import math
import random
# from turtle import Turtle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
# from HFLICorigin.loss import perceptual_loss as ps
# from loss import perceptual_loss as ps
# from HFLICorigin.models.vgg import Vgg16
import lpips

class TVLoss(nn.Module):
    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size(1) * t.size(2) * t.size(3)
    
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metrics = metrics

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metrics == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == 'ms-ssim':
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out
    
def charb(x, alpha, eps):
    '''charbonnier Loss function'''
    return torch.mean(torch.pow(x.pow(2) + eps, 1. / alpha))

def charbnomean(x, alpha, eps):
    '''charbonnier Loss function'''
    return torch.pow(x.pow(2) + eps, 1. / alpha)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, alpha=2,eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target):
        return charb(input - target, self.alpha, self.eps)
    
class CharbonniernomeanLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, alpha=2,eps=1e-6):
        super(CharbonniernomeanLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target):
        return charbnomean(input - target, self.alpha, self.eps)
    
class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type = 'hinge',
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """

        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False, mask=None):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the target is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """

        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        # self.mae = nn.L1Loss()
        self.mae = nn.L1Loss(reduction='none')

    def gram_matrix(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def forward(self,target_feature, inputs):
        G = self.gram_matrix(inputs)
        target = self.gram_matrix(target_feature)
        self.loss = self.mae(G, target)
        return self.loss 

def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target)


def norm(x):
    return x*2-1

def space_to_depth(x, down_scale):
    n,c,h,w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, down_scale, stride=down_scale)
    return unfolded_x.view(n, c* down_scale **2,h // down_scale, w // down_scale)
def get_relaxl1(x):
    # x= norm(x)
    x_d= space_to_depth(x, down_scale=16)
    x_m=torch.mean(x_d,-1)
    x_mean = torch.mean(x_m, -1)
    return x_mean

class ALLLossFaceText(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.TV_loss = TVLoss(weight=0.1)
        self.charbonniernomean = CharbonniernomeanLoss()
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

    def forward(self, output, target, mask_faces, mask_chars):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        x_relax = get_relaxl1(target)
        x_tilde_relax = get_relaxl1(output["x_hat"])

        x_hat_bg = output["x_hat"] * (1 - mask_faces - mask_chars) + target * (mask_faces + mask_chars)
        x_hat_face = output["x_hat"] * mask_faces + target * (1 - mask_faces)
        x_hat_text = output["x_hat"] * mask_chars + target * (1 - mask_chars)

        out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)
        out["charbonnier"] = self.charbonnier(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = self.charbonniernomean(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = torch.mean(out["charbonniernomean"], (1, 2, 3))

        x_pad, target_pad = self.padding_and_trimming(x_hat_bg, target)
        out["lpips"] = self.lpips(x_pad, target_pad, normalize=True)
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        out["mse_loss"] = 150 * self.mse(x_hat_bg, target)
        out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        
        out["face_loss"] = torch.mean((255 * x_hat_face - 255 * target) ** 2, (1, 2, 3))
        out["text_loss"] = torch.mean(self.charbonniernomean(255 * x_hat_text , 255 * target), (1, 2, 3))

        # pool = nn.AvgPool2d(kernel_size=16, stride=16)
        # x_hat_pooled = pool(x_hat_bg)  
        # target_pooled = pool(target)   

        # out["mse_pool_loss"] = self.mse(x_hat_pooled, target_pooled)
        # out["mse_pool_loss"] = torch.mean(out["mse_pool_loss"], (1, 2, 3))

        out["loss"] = out["charbonnier"] + out["lpips"]
        return out

class ALLLossFaceTextHand(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.TV_loss = TVLoss(weight=0.1)
        self.charbonniernomean = CharbonniernomeanLoss()
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

    def forward(self, output, target, mask_faces, mask_chars, mask_hands):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        x_relax = get_relaxl1(target)
        x_tilde_relax = get_relaxl1(output["x_hat"])

        x_hat_bg = output["x_hat"] * (1 - mask_faces - mask_chars - mask_hands) + target * (mask_faces + mask_chars + mask_hands)
        x_hat_face = output["x_hat"] * mask_faces + target * (1 - mask_faces)
        x_hat_text = output["x_hat"] * mask_chars + target * (1 - mask_chars)
        x_hat_hand = output["x_hat"] * mask_hands + target * (1 - mask_hands)

        out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)
        out["charbonnier"] = self.charbonnier(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = self.charbonniernomean(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = torch.mean(out["charbonniernomean"], (1, 2, 3))

        x_pad, target_pad = self.padding_and_trimming(x_hat_bg, target)
        out["lpips"] = self.lpips(x_pad, target_pad, normalize=True)
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        out["mse_loss"] = 150 * self.mse(x_hat_bg, target)
        out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        
        out["face_loss"] = torch.mean((255 * x_hat_face - 255 * target) ** 2, (1, 2, 3))
        out["text_loss"] = torch.mean(self.charbonniernomean(255 * x_hat_text , 255 * target), (1, 2, 3))
        out["hand_loss"] = torch.mean(self.charbonniernomean(255 * x_hat_hand , 255 * target), (1, 2, 3))

        # pool = nn.AvgPool2d(kernel_size=16, stride=16)
        # x_hat_pooled = pool(x_hat_bg)  
        # target_pooled = pool(target)   

        # out["mse_pool_loss"] = self.mse(x_hat_pooled, target_pooled)
        # out["mse_pool_loss"] = torch.mean(out["mse_pool_loss"], (1, 2, 3))

        out["loss"] = out["charbonnier"] + out["lpips"]
        return out
    
class ALLLossFaceTextHandMse(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.TV_loss = TVLoss(weight=0.1)
        self.charbonniernomean = CharbonniernomeanLoss()
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

    def forward(self, output, target, mask_faces, mask_chars, mask_hands):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        x_relax = get_relaxl1(target)
        x_tilde_relax = get_relaxl1(output["x_hat"])

        x_hat_bg = output["x_hat"] * (1 - mask_faces - mask_chars - mask_hands) + target * (mask_faces + mask_chars + mask_hands)
        x_hat_face = output["x_hat"] * mask_faces + target * (1 - mask_faces)
        x_hat_text = output["x_hat"] * mask_chars + target * (1 - mask_chars)
        x_hat_hand = output["x_hat"] * mask_hands + target * (1 - mask_hands)

        out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)
        out["charbonnier"] = self.charbonnier(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = self.charbonniernomean(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = torch.mean(out["charbonniernomean"], (1, 2, 3))

        x_pad, target_pad = self.padding_and_trimming(x_hat_bg, target)
        out["lpips"] = self.lpips(x_pad, target_pad, normalize=True)
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        out["mse_loss"] = 150 * self.mse(x_hat_bg, target)
        out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        
        out["face_loss"] = torch.mean((255 * x_hat_face - 255 * target) ** 2, (1, 2, 3))
        out["text_loss"] = torch.mean(self.charbonniernomean(255 * x_hat_text , 255 * target), (1, 2, 3))
        out["hand_loss"] = torch.mean((255 * x_hat_hand - 255 * target) ** 2, (1, 2, 3))

        # pool = nn.AvgPool2d(kernel_size=16, stride=16)
        # x_hat_pooled = pool(x_hat_bg)  
        # target_pooled = pool(target)   

        # out["mse_pool_loss"] = self.mse(x_hat_pooled, target_pooled)
        # out["mse_pool_loss"] = torch.mean(out["mse_pool_loss"], (1, 2, 3))

        out["loss"] = out["charbonnier"] + out["lpips"]
        return out

class ALLLossFaceText2(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.charbonniernomean = CharbonniernomeanLoss()
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

    def forward(self, output, rec, target, mask_faces, mask_chars):
        N, _, H, W = target.size()
        H,W = rec.size(2), rec.size(3)
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        x_relax = get_relaxl1(target)
        x_tilde_relax = get_relaxl1(rec)
        
        x_hat_bg = rec * (1 - mask_faces - mask_chars) + target * (mask_faces + mask_chars)
        x_hat_face = rec * mask_faces + target * (1 - mask_faces)
        x_hat_text = rec * mask_chars + target * (1 - mask_chars)

        out["charbonnier"] = self.charbonnier(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = self.charbonniernomean(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = torch.mean(out["charbonniernomean"], (1, 2, 3))
        
        out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)
        
        x_pad, target_pad = self.padding_and_trimming(x_hat_bg, target)
        out["lpips"] = self.lpips(x_pad, target_pad, normalize=True)
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        out["mse_loss"] = 150 * self.mse(x_hat_bg, target)
        out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        
        out["face_loss"] = torch.mean((255 * x_hat_face - 255 * target) ** 2, (1, 2, 3))
        out["text_loss"] = torch.mean(self.charbonniernomean(255 * x_hat_text , 255 * target), (1, 2, 3))

        out["loss"] = out["charbonnier"] + out["lpips"]
        return out

class ALLLossFaceText3(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.charbonniernomean = CharbonniernomeanLoss()
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

    def forward(self, output, rec, target, mask_faces, mask_chars):
        N, _, H, W = target.size()
        H,W = rec.size(2), rec.size(3)
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        x_relax = get_relaxl1(target)
        x_tilde_relax = get_relaxl1(rec)
        
        x_hat_bg = rec * (1 - mask_faces - mask_chars) + target * (mask_faces + mask_chars)
        x_hat_face = rec * mask_faces + target * (1 - mask_faces)
        x_hat_text = rec * mask_chars + target * (1 - mask_chars)

        out["charbonnier"] = self.charbonnier(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = self.charbonniernomean(255 * x_hat_bg, 255 * target)
        out["charbonniernomean"] = torch.mean(out["charbonniernomean"], (1, 2, 3))
        
        out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)
        
        x_pad, target_pad = self.padding_and_trimming(x_hat_bg, target)
        out["lpips"] = self.lpips(x_pad, target_pad, normalize=True)
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        out["mse_loss"] = 150 * self.mse(x_hat_bg, target)
        out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        
        out["face_loss"] = torch.mean((255 * x_hat_face - 255 * target) ** 2, (1, 2, 3))
        out["text_loss"] = torch.mean(self.charbonniernomean(255 * x_hat_text , 255 * target), (1, 2, 3))

        out["loss"] = out["charbonnier"] + out["lpips"]
        return out
    #text loss + face loss , 外加一个权重
            # out["bpp_loss_obj"] = sum(
        #     (torch.log(likelihoods * mask).sum() / (-math.log(2) * num_pixels))
        #     for likelihoods in output["likelihoods"].values()
        # )

        # out["bpp_loss_bg"] = sum(
        #     (torch.log(likelihoods * (1 - mask)).sum() / (-math.log(2) * num_pixels))
        #     for likelihoods in output["likelihoods"].values()
        # )

                # x_hat_feat  = self.vgg(output["x_hat"])
        # target_feat = self.vgg(target)
        # x_hat_feat  = [feat for feat in  x_hat_feat]
        # target_feat = [feat for feat in  target_feat]
        # style_loss = 0
        # for i in range(4):
        #     style_loss += torch.mean(self.style(x_hat_feat[i], target_feat[i]))
        # out["style_loss"]  = style_loss