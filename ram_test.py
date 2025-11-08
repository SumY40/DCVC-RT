# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 限定使用第0号GPU
import torch

from src.models.video_model_gan import DMC
from src.models.image_model import DMCI
from src.utils.common import get_state_dict \




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

i_frame_net = DMCI()
# i_state_dict = get_state_dict('./ckpts/images/yuv_input/gan/tempbest_image.pth')
# i_frame_net.load_state_dict(i_state_dict)
i_frame_net = i_frame_net.to(device)
# i_frame_net.eval()
# i_frame_net.update()
# i_frame_net.half()

p_frame_net = DMC()
# p_state_dict = get_state_dict('./ckpts/yuv_input/gan/patchgan/tempbest.pth')
# p_frame_net.load_state_dict(p_state_dict)
p_frame_net = p_frame_net.to(device)
# p_frame_net.eval()
# p_frame_net.update()
# p_frame_net.half()
# # 模型加载和配置完成
torch.cuda.empty_cache()

print("模型加载完成，暂停 30 秒...")
time.sleep(30)
ddd 