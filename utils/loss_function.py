from pytorch_msssim import ms_ssim as MS_SSIM
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as tnf
import lpips
mse = torch.nn.MSELoss().to(device)
import torch.nn as nn
import random
import torch.nn.functional as F
from src.utils.transforms import *
from utils.mask import get_face_mask

def msssim_loss(fake, real):
    assert fake.shape == real.shape
    return 1 - MS_SSIM(fake, real, data_range=1.0, size_average=False)
class MSELossYUV2YUV(nn.Module):
    r"""Input RGB, calculate MSE loss in YUV format.
    """
    def __init__(self, gamma=2):
        super(MSELossYUV2YUV, self).__init__()
        self.gamma = gamma
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, img1, img2, improve_uv=1, norm=False, type='bilinear'):
        r"""
        Args: 
            img1: First RGB Image.
            img2: Second RGB Image.

        Returns:
            distortion_loss : Loss of MSE in YUV
            Y_distortion    : PSNR in Y
            distortion      : PSNR in YUV
        """
        # rgb to yuv444
        img_yuv1 = torch.clamp(img1, 0, 1)
        img_yuv2 = torch.clamp(img2, 0, 1)
        y1, u1, v1 = img_yuv1.split([1,1,1], dim=1)
        y2, u2, v2 = img_yuv2.split([1,1,1], dim=1)
        # yuv444 to yuv420
        #print(u1.shape)
        if type == 'nearest':
            u1 = u1[:,:,::2,::2]
            u2 = u2[:,:,::2,::2]
            v1 = v1[:,:,::2,::2]
            v2 = v2[:,:,::2,::2]
        else:
            b,_,h,w = u1.shape
            u1 = torch.mean(u1.reshape(b, h//2, 2, w//2, 2), axis=(-1, -3))
            u2 = torch.mean(u2.reshape(b, h//2, 2, w//2, 2), axis=(-1, -3))
            v1 = torch.mean(v1.reshape(b, h//2, 2, w//2, 2), axis=(-1, -3))
            v2 = torch.mean(v2.reshape(b, h//2, 2, w//2, 2), axis=(-1, -3))

        # calculate loss
        Y_loss = torch.mean(self.loss(y1, y2),dim=(1,2,3))
        U_loss = torch.mean(self.loss(u1, u2),dim=(1,2))
        V_loss = torch.mean(self.loss(v1, v2),dim=(1,2))
        # Y_distortion = 10 * torch.log10(1.0 / Y_loss).item()
        # U_distortion = 10 * torch.log10(1.0 / U_loss).item()
        # V_distortion = 10 * torch.log10(1.0 / V_loss).item()
        distortion_loss = (self.gamma * Y_loss + improve_uv * (U_loss +
                           V_loss)) / (self.gamma + 2)
        # return Y_loss,U_loss,V_loss
        return distortion_loss #, np.array([Y_distortion, U_distortion, V_distortion])


def mse_RGB_loss(dec, gt):
    assert gt.shape == dec.shape
    sum_loss = tnf.mse_loss(dec, gt, reduction='none').mean(dim=(1, 2, 3))
    return sum_loss

def charbnomean(x, alpha, eps):
    '''charbonnier Loss function'''
    return torch.pow(x.pow(2) + eps, 1. / alpha)
# def charbnomean(x, eps):
    # return torch.sqrt(x.pow(2) + eps**2)


def norm(x):
    return x*2-1

def space_to_depth(x, down_scale):
    n,c,h,w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, down_scale, stride=down_scale)
    return unfolded_x.view(n, c* down_scale **2,h // down_scale, w // down_scale)
def get_relaxl1(x):
    x= norm(x)
    x_d= space_to_depth(x, down_scale=16)
    x_m=torch.mean(x_d,-1)
    x_mean = torch.mean(x_m, -1)
    return x_mean

# class CharbonniernomeanLoss(torch.nn.Module):
#     """标准 Charbonnier Loss"""
#     def __init__(self, eps=1e-6):
#         super(CharbonniernomeanLoss, self).__init__()
#         self.eps = eps

#     def forward(self, input, target):
#         return charbnomean(input - target, self.eps)


class CharbonniernomeanLoss(torch.nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, alpha=2,eps=1e-6):
        super(CharbonniernomeanLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target):
        return charbnomean(input - target, self.alpha, self.eps)

def periodic_compensation_loss(
    orig: torch.Tensor,
    recon: torch.Tensor,
    period: int
) -> torch.Tensor:
    B, C, H, W = orig.shape
    if H % period != 0 or W % period != 0:
        raise ValueError("Height and Width must be divisible by period.")
    
    # 1. 提取块：结果是 [B, C, H//p, W//p, p, p]
    patches_orig  = orig.unfold(2, period, period).unfold(3, period, period)
    patches_recon = recon.unfold(2, period, period).unfold(3, period, period)
    
    # 2. 重排：把 [H//p, W//p, p, p] => [H, W]
    #    最终目的是对每个周期内像素位置做平均
    #    即得到 [B, C, p, p]：checkerboard pattern 的周期图像
    B_orig = patches_orig.permute(0,1,4,5,2,3).reshape(B, C, period, period, -1).mean(dim=-1)
    B_recon = patches_recon.permute(0,1,4,5,2,3).reshape(B, C, period, period, -1).mean(dim=-1)

    # 3. 计算周期图像的 MSE
    loss = F.mse_loss(B_orig, B_recon, reduction='none').mean(dim=(1,2,3))
    return loss

from DISTS_pytorch import DISTS
class ALLDistsLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        self.charbonniernomean = CharbonniernomeanLoss()
        self.dists_metric = DISTS().cuda()
        
        self.lmbda = lmbda
        self.l1 = nn.L1Loss(reduction='none')
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

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        rec =  output["x_hat"]
        rec_rgb = ycbcr2rgb(rec)
        target_rgb = ycbcr2rgb(target)
        # x_relax = get_relaxl1(target_rgb)
        # x_tilde_relax = get_relaxl1(rec_rgb)


        # out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)

        out["l1_yuv"] = torch.mean(self.l1(255 * rec, 255 * target), (1, 2, 3))
        out["l1_rgb"] = torch.mean(self.l1(255 * rec_rgb, 255 * target_rgb), (1, 2, 3))
        out["l1"] = 0.8 * out["l1_yuv"] + 0.2 * out["l1_rgb"]

        out["l1_yuv_01"] = torch.mean(self.l1(rec, target), (1, 2, 3))
        out["l1_rgb_01"] = torch.mean(self.l1(rec_rgb, target_rgb), (1, 2, 3))
        out["l1_01"] = 0.8 * out["l1_yuv_01"] + 0.2 * out["l1_rgb_01"]

        out["l2_yuv_01"] = torch.mean(self.mse(rec, target), (1, 2, 3))
        out["l2_rgb_01"] = torch.mean(self.mse(rec_rgb, target_rgb), (1, 2, 3))
        out["l2_01"] = 0.8 * out["l2_yuv_01"] + 0.2 * out["l2_rgb_01"]


        x_pad, target_pad = self.padding_and_trimming(rec, target)
        out["dists"] = self.dists_metric(rec_rgb, target_rgb, require_grad= True, batch_average=False)
        # 输入01
        out["artifact"] = periodic_compensation_loss(255 * rec_rgb, 255 * target_rgb, period=8)
        out["artifact_01"] = periodic_compensation_loss(rec_rgb, target_rgb, period=8)

        # #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        # out["mse_loss"] = 150 * self.mse(rec, target)
        # out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        

        return out

class ALLLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
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

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        rec =  output["x_hat"]
        x_relax = get_relaxl1(target)
        x_tilde_relax = get_relaxl1(output["x_hat"])


        out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)
        out["charbonniernomean"] = self.charbonniernomean(255 * rec, 255 * target)
        out["charbonniernomean"] = torch.mean(out["charbonniernomean"], (1, 2, 3))

        x_pad, target_pad = self.padding_and_trimming(rec, target)
        out["lpips"] = self.lpips(x_pad, target_pad, normalize=True)
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        out["mse_loss"] = 150 * self.mse(rec, target)
        out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        

        return out
    
class ALLYUVLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        self.charbonniernomean = CharbonniernomeanLoss()
        self.lpips = lpips.LPIPS(net='vgg').cuda()
        
        self.lmbda = lmbda
        self.l1 = nn.L1Loss(reduction='none')
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

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        rec =  output["x_hat"]
        rec_rgb = ycbcr2rgb(rec)
        target_rgb = ycbcr2rgb(target)
        # x_relax = get_relaxl1(target_rgb)
        # x_tilde_relax = get_relaxl1(rec_rgb)


        # out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)

        out["l1_yuv"] = torch.mean(self.l1(255 * rec, 255 * target), (1, 2, 3))
        out["l1_rgb"] = torch.mean(self.l1(255 * rec_rgb, 255 * target_rgb), (1, 2, 3))
        out["l1"] = 0.8 * out["l1_yuv"] + 0.2 * out["l1_rgb"]

        out["l1_yuv_01"] = torch.mean(self.l1(rec, target), (1, 2, 3))
        out["l1_rgb_01"] = torch.mean(self.l1(rec_rgb, target_rgb), (1, 2, 3))
        out["l1_01"] = 0.8 * out["l1_yuv_01"] + 0.2 * out["l1_rgb_01"]

        out["l2_yuv_01"] = torch.mean(self.mse(rec, target), (1, 2, 3))
        out["l2_rgb_01"] = torch.mean(self.mse(rec_rgb, target_rgb), (1, 2, 3))
        out["l2_01"] = 0.8 * out["l2_yuv_01"] + 0.2 * out["l2_rgb_01"]


        x_pad, target_pad = self.padding_and_trimming(rec, target)
        out["lpips"] = self.lpips(rec_rgb, target_rgb, normalize=True) # 输入0-1
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        out["artifact"] = periodic_compensation_loss(255 * rec_rgb, 255 * target_rgb, period=8)
        out["artifact_01"] = periodic_compensation_loss(rec_rgb, target_rgb, period=8)

        # #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        # out["mse_loss"] = 150 * self.mse(rec, target)
        # out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        

        return out

class ALLMaskYUVLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
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

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        rec =  output["x_hat"]
        rec_rgb = ycbcr2rgb(rec)
        target_rgb = ycbcr2rgb(target)
        # x_relax = get_relaxl1(target_rgb)
        # x_tilde_relax = get_relaxl1(rec_rgb)

        mask_faces = get_face_mask(target_rgb)
        x_hat_bg_rgb = rec_rgb * (1 - mask_faces) + target_rgb * (mask_faces )
        x_hat_face_rgb =  rec_rgb * mask_faces  + target_rgb * (1 - mask_faces) # 背景换成gt

        # out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)
        out["charbonniernomean_yuv"] = self.charbonniernomean(255 * rec, 255 * target)
        out["charbonniernomean_yuv"] = torch.mean(out["charbonniernomean_yuv"], (1, 2, 3))

        out["charbonniernomean_rgb"] = self.charbonniernomean(255 * rec_rgb, 255 * target_rgb)
        out["charbonniernomean_rgb"] = torch.mean(out["charbonniernomean_rgb"], (1, 2, 3))
        out["charbonniernomean"] = 0.8 * out["charbonniernomean_yuv"] + 0.2 * out["charbonniernomean_rgb"]


        out["lpips"] = self.lpips(rec_rgb, target_rgb, normalize=True)
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        out["artifact"] = periodic_compensation_loss(255 * rec_rgb, 255 * target_rgb, period=8)
        # out["face_loss"] = torch.mean((255 * x_hat_face_rgb - 255 * target_rgb) ** 2, (1, 2, 3))
        out["face_loss"] = self.mse(255 * x_hat_face_rgb, 255 * target_rgb).mean(dim=(1, 2, 3))
        # out["charbonniernomean_face"] = self.charbonniernomean(255 * x_hat_face_rgb, 255 * target_rgb)
        # out["charbonniernomean_face"] = torch.mean(out["charbonniernomean_face"], (1, 2, 3))

        # #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        # out["mse_loss"] = 150 * self.mse(rec, target)
        # out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        

        return out



class ALLYUVDistsLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        self.charbonniernomean = CharbonniernomeanLoss()
        self.lpips = lpips.LPIPS(net='alex').cuda()
        self.dists_metric = DISTS().to(device)
        self.lmbda = lmbda
        self.l1 = nn.L1Loss(reduction='none')
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

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        rec =  output["x_hat"]
        rec_rgb = ycbcr2rgb(rec)
        target_rgb = ycbcr2rgb(target)
        # x_relax = get_relaxl1(target_rgb)
        # x_tilde_relax = get_relaxl1(rec_rgb)


        # out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)

        out["l1_yuv"] = torch.mean(self.l1(255 * rec, 255 * target), (1, 2, 3))
        out["l1_rgb"] = torch.mean(self.l1(255 * rec_rgb, 255 * target_rgb), (1, 2, 3))
        out["l1"] = 0.8 * out["l1_yuv"] + 0.2 * out["l1_rgb"]

        out["l1_yuv_01"] = torch.mean(self.l1(rec, target), (1, 2, 3))
        out["l1_rgb_01"] = torch.mean(self.l1(rec_rgb, target_rgb), (1, 2, 3))
        out["l1_01"] = 0.8 * out["l1_yuv_01"] + 0.2 * out["l1_rgb_01"]

        out["l2_yuv_01"] = torch.mean(self.mse(rec, target), (1, 2, 3))
        out["l2_rgb_01"] = torch.mean(self.mse(rec_rgb, target_rgb), (1, 2, 3))
        out["l2_01"] = 0.8 * out["l2_yuv_01"] + 0.2 * out["l2_rgb_01"]


        x_pad, target_pad = self.padding_and_trimming(rec, target)
        out["lpips"] = self.lpips(rec_rgb, target_rgb, normalize=True) # 输入0-1
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        out["artifact"] = periodic_compensation_loss(255 * rec_rgb, 255 * target_rgb, period=8)
        out["artifact_01"] = periodic_compensation_loss(rec_rgb, target_rgb, period=8)

        out["dists"] = self.dists_metric(rec_rgb, target_rgb, require_grad= True, batch_average=False)
        # #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        # out["mse_loss"] = 150 * self.mse(rec, target)
        # out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        

        return out

class ALLRGBDistsLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, device="cuda", gpu_id=None):
        super().__init__()
        self.charbonniernomean = CharbonniernomeanLoss()
        self.lpips = lpips.LPIPS(net='alex').cuda()
        self.dists_metric = DISTS().to(device)
        self.lmbda = lmbda
        self.l1 = nn.L1Loss(reduction='none')
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

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        rec =  output["x_hat"]

        # x_relax = get_relaxl1(target_rgb)
        # x_tilde_relax = get_relaxl1(rec_rgb)


        # out["artifact"] = torch.mean(self.l1(x_relax, x_tilde_relax), 1)
        # out["tv_loss"] = self.TV_loss(x_hat_bg)

        out["l1_rgb"] = torch.mean(self.l1(255 * rec, 255 * target), (1, 2, 3))
        out["l1"] = out["l1_rgb"]


        out["l1_rgb_01"] = torch.mean(self.l1(rec, target), (1, 2, 3))
        out["l1_01"] = out["l1_rgb_01"]


        out["l2_rgb_01"] = torch.mean(self.mse(rec, target), (1, 2, 3))
        out["l2_01"] =out["l2_rgb_01"]


        x_pad, target_pad = self.padding_and_trimming(rec, target)
        out["lpips"] = self.lpips(rec, target, normalize=True) # 输入0-1
        out["lpipsnomean"] = torch.mean(out["lpips"], (1, 2, 3))

        out["artifact"] = periodic_compensation_loss(255 * rec, 255 * target, period=8)
        out["artifact_01"] = periodic_compensation_loss(rec, target, period=8)

        out["dists"] = self.dists_metric(rec, target, require_grad= True, batch_average=False)
        # #out["mse_loss"] = torch.mean(self.mse(255 * x_hat_bg, 255 * target))
        # out["mse_loss"] = 150 * self.mse(rec, target)
        # out["mse_lossnomean"] = torch.mean(out["mse_loss"],(1,2,3))
        

        return out