import torch
import os
from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.utils.common import get_state_dict

# 初始化并加载 FP16 的 i-frame 模型
global i_frame_net
i_frame_net = DMCI()
i_state_dict = get_state_dict('./checkpoints/cvpr2025_image.pth.tar')
i_frame_net.load_state_dict(i_state_dict)
i_frame_net = i_frame_net.half()  # 转 FP16

# 初始化并加载 FP16 的 p-frame 模型
global p_frame_net
p_frame_net = DMC()
p_state_dict = get_state_dict('./checkpoints/cvpr2025_video.pth.tar')
p_frame_net.load_state_dict(p_state_dict)
p_frame_net = p_frame_net.half()  # 转 FP16

# 保存为 .pth（state_dict形式）
torch.save(i_frame_net.state_dict(), './checkpoints/i_frame_fp16.pth')
torch.save(p_frame_net.state_dict(), './checkpoints/p_frame_fp16.pth')

# 打印文件大小（单位：MB）
i_size = os.path.getsize('./checkpoints/i_frame_fp16.pth') / 1024 / 1024
p_size = os.path.getsize('./checkpoints/p_frame_fp16.pth') / 1024 / 1024
print(f"i_net size: {i_size:.2f} MB, p_net size: {p_size:.2f} MB")
