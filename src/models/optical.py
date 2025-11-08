import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function



class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.me_8x = MEBasic(0)
        self.me_4x = MEBasic(0)
        self.me_2x = MEBasic(3)
        self.me_1x = MEBasic(3)

    def forward(self, im1, im2):
        batchsize = im1.size()[0]

        im1_1x = im1
        im1_2x = F.avg_pool2d(im1_1x, kernel_size=2, stride=2)
        im1_4x = F.avg_pool2d(im1_2x, kernel_size=2, stride=2)
        im1_8x = F.avg_pool2d(im1_4x, kernel_size=2, stride=2)
        im2_1x = im2
        im2_2x = F.avg_pool2d(im2_1x, kernel_size=2, stride=2)
        im2_4x = F.avg_pool2d(im2_2x, kernel_size=2, stride=2)
        im2_8x = F.avg_pool2d(im2_4x, kernel_size=2, stride=2)

        shape_fine = im1_8x.size()
        zero_shape = [batchsize, 2, shape_fine[2], shape_fine[3]]
        flow_8x = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        flow_8x = self.me_8x(torch.cat((im1_8x, im2_8x, flow_8x), dim=1))

        flow_4x = bilinearupsacling(flow_8x) * 2.0
        flow_4x = flow_4x + self.me_4x(torch.cat((im1_4x,
                                                  block_mc_func(im2_4x, flow_4x),
                                                  flow_4x),
                                                 dim=1))

        flow_2x = bilinearupsacling(flow_4x) * 2.0
        flow_2x = flow_2x + self.me_2x(torch.cat((im1_2x,
                                                  block_mc_func(im2_2x, flow_2x),
                                                  flow_2x),
                                                 dim=1))

        flow_1x = bilinearupsacling(flow_2x) * 2.0
        flow_1x = flow_1x + self.me_1x(torch.cat((im1_1x,
                                                  block_mc_func(im2_1x, flow_1x),
                                                  flow_1x),
                                                 dim=1))
        return flow_1x

class MEBasic(nn.Module):
    def __init__(self, complexity_level=0):
        super().__init__()
        self.relu = nn.ReLU()
        self.by_pass = False
        if complexity_level < 0:
            self.by_pass = True
        elif complexity_level == 0:
            self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
            self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
            self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
            self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
            self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        elif complexity_level == 3:
            self.conv1 = nn.Conv2d(8, 32, 5, 1, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
            self.conv3 = nn.Conv2d(64, 32, 5, 1, padding=2)
            self.conv4 = nn.Conv2d(32, 16, 5, 1, padding=2)
            self.conv5 = nn.Conv2d(16, 2, 5, 1, padding=2)

    def forward(self, x):
        if self.by_pass:
            return x[:, -2:, :, :]

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x
    

def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)

    return outfeature


def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature

def torch_warp(feature, flow):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    add_grid_cache(flow)
    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = backward_grid[device_id][str(flow.size())] + flow
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


def set_force_recalculate_grid(force):
    global FORCE_RECALCULATE_GRID
    FORCE_RECALCULATE_GRID = force
backward_grid = [{} for _ in range(9)]    # 0~7 for GPU, -1 for CPU
FORCE_RECALCULATE_GRID = False

def add_grid_cache(flow):
    device_id = -1 if flow.device == torch.device('cpu') else flow.device.index
    if str(flow.size()) not in backward_grid[device_id] or FORCE_RECALCULATE_GRID:
        B, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=flow.device, dtype=torch.float32).view(
            1, 1, 1, W).expand(B, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=flow.device, dtype=torch.float32).view(
            1, 1, H, 1).expand(B, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)


def flow_warp(im, flow):
    is_float16 = False
    if im.dtype == torch.float16:
        is_float16 = True
        im = im.to(torch.float32)
        flow = flow.to(torch.float32)
    warp = torch_warp(im, flow)
    if is_float16:
        warp = warp.to(torch.float16)
    return warp

def block_mc_func(im, flow):
    return flow_warp(im, flow)
