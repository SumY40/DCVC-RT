import io
import os
import math
import glob
import time
import numpy as np
import torch
import torch.nn.functional as F
from src.utils.common import str2bool, create_folder, generate_log_json, get_state_dict, generate_log_json_lpips \

from PIL import Image
from torchvision import transforms
from src.layers.cuda_inference import replicate_pad
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np

from src.models.image_model import DMCI
from PIL import Image
import argparse
from src.utils.stream_helper import SPSHelper, NalType, write_sps, read_header, \
    read_sps_remaining, read_ip_remaining, write_ip
def parse_args_testing(input_args=None):

    parser = argparse.ArgumentParser()

    # pretrained weights
    parser.add_argument("--sd_path", help="path to SD-Turbo")
    parser.add_argument("--elic_path", help="path to pretrained ELIC model")
    parser.add_argument("--codec_path", help="path to pretrained StableCodec weights", default=None)
    parser.add_argument('--model_path_i', type=str)

    parser.add_argument('--pretrained_weights', type=str, default="",
                        help='')
    # testing images
    parser.add_argument("--img_path", type=str, default='/data/Kodak/')

    # output path
    parser.add_argument("--rec_path", type=str, default='./out_bin/intra/rec/')
    parser.add_argument("--bin_path", type=str, default='./out_bin/intra/bin/')
    parser.add_argument("--qp_i", type=int, default=10)

    # testing details
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument('--local_rank', default=-1, type=int)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor

def compress_one_image(net, bin_path, ori_h, ori_w, img_name, x):
    padding_r, padding_b = DMCI.get_padding_size(ori_h, ori_w, 16)

    torch.cuda.synchronize()
    frame_start_time = time.time()
    x_padded = replicate_pad(x, padding_b, padding_r)
    sps_helper = SPSHelper()
    curr_qp = args.qp_i
    output_buff = io.BytesIO()
    use_two_entropy_coders = ori_h * ori_w > 1280 * 720
    net.set_use_two_entropy_coders(use_two_entropy_coders)
    with torch.no_grad():
        encoded = net.compress(x_padded, args.qp_i)
    sps = {
        'sps_id': -1,
        'height': ori_h,
        'width': ori_w,
        'ec_part': 1 if use_two_entropy_coders else 0,
        'use_ada_i': 0,
    }


    sps_id, sps_new = sps_helper.get_sps_id(sps)
    sps['sps_id'] = sps_id
    sps_bytes = 0
    if sps_new:
        sps_bytes = write_sps(output_buff, sps)

    is_i_frame = True

    stream_bytes = write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded['bit_stream'])
    with open(bin_path, "wb") as output_file:
        bytes_buffer = output_buff.getbuffer()
        output_file.write(bytes_buffer)
        total_bytes = bytes_buffer.nbytes
        bytes_buffer.release()
    output_buff.close()
    torch.cuda.synchronize()
    frame_end_time = time.time()
    bits = stream_bytes * 8 + sps_bytes * 8
    bpp = bits / (ori_h * ori_w)
    # _,_, total_bits = net.forward_eval(x, [1])
    # bpp = float(total_bits) / (ori_h * ori_w)
    frame_time = frame_end_time - frame_start_time

    return bpp, frame_time

def decompress_one_image(net, bin_path, ori_h, ori_w, img_name):
    sps_helper = SPSHelper()
    input_file = open(bin_path, "rb")
    with open(bin_path, "rb") as input_file:
        input_buff = io.BytesIO(input_file.read())
    torch.cuda.synchronize()
    frame_start_time = time.time()
    header = read_header(input_buff)
    while header['nal_type'] == NalType.NAL_SPS:
        sps = read_sps_remaining(input_buff, header['sps_id'])
        sps_helper.add_sps_by_id(sps)
        header = read_header(input_buff)
        continue
    sps_id = header['sps_id']

    sps = sps_helper.get_sps_by_id(sps_id)
    qp, bit_stream = read_ip_remaining(input_buff)

    output = os.path.join(bin_path, img_name)
    with torch.no_grad():
        # out_img = net.decompress(strings, shape, prompt)
        decoded = net.decompress(bit_stream, sps, qp)
        recon_frame = decoded['x_hat']
    out_img = recon_frame[:, :, :ori_h, :ori_w].float().cpu().detach()
    out_img = ycbcr2rgb(out_img)
    torch.cuda.synchronize()
    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time

    return out_img, frame_time

def main(args):

    global i_frame_net
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.cuda()
    i_frame_net.eval()
    i_frame_net.update(None)
    i_frame_net.half()


    transform = transforms.Compose([
        # transforms.Resize((256, 384)),
        transforms.ToTensor(),
    ])
      
    bpp = []
    pos_tag_prompt = [1]
    images = glob.glob(args.img_path + '/*.png')
    print(f'\nFind {str(len(images))} images in {args.img_path}\n')
    total_enctime = 0
    total_dectime = 0


    for idx_img, img_path in enumerate(images):

        print('[Processing]', img_path)
        (path, name) = os.path.split(img_path)
        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.rec_path, fname+'.png')

        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        img_in = rgb2ycbcr(img)
        img_in = img_in.to(torch.float16)

        ori_h, ori_w = img.shape[2:]


        curr_bin_path = os.path.join(args.bin_path, fname + '.bin')
        qp = 20
        if idx_img == 0:
            _, _ = compress_one_image(i_frame_net, curr_bin_path, ori_h, ori_w, fname, img_in)
            
        with torch.no_grad():
            try:
                rate, enc_time = compress_one_image(i_frame_net, curr_bin_path, ori_h, ori_w, fname, img_in)
                out_img, dec_time = decompress_one_image(i_frame_net, curr_bin_path, ori_h, ori_w, fname)
                # out = net(img_padded, qp)
                # out_img = out['x_hat']
                # bpp_y = out['bpp_y']
                # bpp_z = out['bpp_z']
                # out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
                # out_img = out_img .float().cpu().detach()
                # rate = (bpp_y + bpp_z).mean().item()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(str(name))
                    print("CUDA out of memory. Continuing to next image.")
                    torch.cuda.empty_cache() 
                    continue
                else:
                    raise

        output_pil = transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0))

        bpp.append(rate)
        print('[BPP]', rate)
        print('[enc_time]', enc_time)  
        print('[dec_time]', dec_time)
        total_dectime += dec_time
        total_enctime += enc_time

        output_pil.save(outf)

    print('\n[Average BPP]', np.mean(bpp))
    print('\n[Average enctime]', total_enctime / len(images))
    print('\n[Average dectime]', total_dectime / len(images))

if __name__ == "__main__":
    args = parse_args_testing()
    rec_path = './out_bin/intra/'
    if not os.path.exists(args.rec_path): os.makedirs(args.rec_path)
    if not os.path.exists(args.bin_path): os.makedirs(args.bin_path)
    main(args)
