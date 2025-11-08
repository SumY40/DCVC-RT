# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from src.layers.cuda_inference import replicate_pad
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.common import str2bool, create_folder, generate_log_json, get_state_dict, \
    dump_json, set_torch_env
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
i_frame_net = DMCI()
i_state_dict = get_state_dict('./ckpts/images/yuv_input/gan/rt_intra_32_RD_1_epoch_17iter_64000.pth')
i_frame_net.load_state_dict(i_state_dict)
i_frame_net = i_frame_net.to(device)
i_frame_net.eval()
i_frame_net.update()
transform = transforms.Compose([
    transforms.ToTensor(),  # 把 [0,255] 的 PIL 图像转为 [0,1] 的 Tensor
])
def load_image(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    return image


img = load_image('./test/raw.bmp')
x = transform(img).unsqueeze(0).to(device)

x = rgb2ycbcr(x)
pic_height = 2166
pic_width = 250
qp = 0
padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)
x_padded = replicate_pad(x, padding_b, padding_r)
print(f"x_padded shape: {x_padded.shape}")

curr_qp = DMC.get_q_tensor(qp, device)
result = i_frame_net(x_padded, curr_qp )
bit = result["bit"].cpu().item()

recon_frame = ycbcr2rgb(result["x_hat"])
recon_frame = recon_frame.clamp_(0, 1)
x_hat = recon_frame[:, :, :pic_height, :pic_width]
# 将值域 [0,1] → [0,255] 并转换成 uint8，然后保存
output_img = to_pil_image(x_hat.squeeze(0).cpu())  # squeeze 是因为有 batch 维度
print("bpp", bit/ (pic_height * pic_width))
output_img.save("output.bmp")