# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from src.models.optical import ME_Spynet, block_mc_func

import argparse
import concurrent.futures
import io
import json
import multiprocessing
import os
import time

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from src.layers.cuda_inference import replicate_pad
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.common import str2bool, create_folder, get_state_dict, \
    dump_json, set_torch_env
from src.utils.stream_helper import SPSHelper, NalType, write_sps, read_header, \
    read_sps_remaining, read_ip_remaining, write_ip
from src.utils.video_reader import PNGReader, YUV420Reader
from src.utils.video_writer import PNGWriter, YUV420Writer
from src.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np, ycbcr420_to_rgb_np


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--force_zero_thres', type=float, default=None, required=False)
    parser.add_argument('--model_path',  type=str)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--qp_i', type=int, nargs="+")
    parser.add_argument('--qp_p', type=int, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--reset_interval', type=int, default=32, required=False)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, default=False)
    parser.add_argument('--cuda_idx', type=int, nargs="+", help='GPU indexes to use')
    parser.add_argument('--calc_ssim', type=str2bool, default=False, required=False)
    parser.add_argument('--write_stream', type=str2bool, default=False)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose_json', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--run_fast', type=str2bool, default=False)

    args = parser.parse_args()
    return args

def generate_log_json(frame_num, flows, inter_psnrs, hist_diffs):
    p_num = 0
    flow = 0
    hist_diff = 0
    inter_psnr = 0
    for idx in range(frame_num):
        flow += flows[idx]
        hist_diff += hist_diffs[idx]
        inter_psnr += inter_psnrs[idx]

    log_result = {}



    log_result['ave_all_flow'] = flow / frame_num
    log_result['ave_all_inter_psnr'] = inter_psnr / frame_num
    log_result['ave_all_hist_diff'] = hist_diff / frame_num



    return log_result


def np_image_to_tensor(img, device):
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image


def get_src_reader(args):

    if args['src_type'] == 'png':
        src_reader = PNGReader(args['src_path'], args['src_width'], args['src_height'])
    elif args['src_type'] == 'yuv420':
        src_reader = YUV420Reader(args['src_path'], args['src_width'], args['src_height'])
    return src_reader


def get_src_frame(args, src_reader, device):
    if args['src_type'] == 'yuv420':
        y, uv = src_reader.read_one_frame()
        yuv = ycbcr420_to_444_np(y, uv)
        x_yuv = np_image_to_tensor(yuv, device)
        y = y[0, :, :]
        u = uv[0, :, :]
        v = uv[1, :, :]
        rgb = ycbcr420_to_rgb_np(y,uv) * 255
        x_rgb = ycbcr2rgb(x_yuv)
    else:
        assert args['src_type'] == 'png'
        rgb = src_reader.read_one_frame()
        x_rgb = np_image_to_tensor(rgb, device)
        x_yuv = rgb2ycbcr(x_rgb)
        y, u, v = None, None, None

    return x_yuv, y, u, v, x_rgb, rgb



def get_distortion(args, x_hat, y, u, v, rgb):
    if args['src_type'] == 'yuv420':
        y_rec, uv_rec = yuv_444_to_420(x_hat)
        y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        y_rec = y_rec[0, :, :]
        u_rec = uv_rec[0, :, :]
        v_rec = uv_rec[1, :, :]
        psnr_y = calc_psnr(y, y_rec)
        psnr_u = calc_psnr(u, u_rec)
        psnr_v = calc_psnr(v, v_rec)
        psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
        if args['calc_ssim']:
            ssim_y = calc_msssim(y, y_rec)
            ssim_u = calc_msssim(u, u_rec)
            ssim_v = calc_msssim(v, v_rec)
        else:
            ssim_y, ssim_u, ssim_v = 0., 0., 0.
        ssim = (6 * ssim_y + ssim_u + ssim_v) / 8

        curr_psnr = [psnr, psnr_y, psnr_u, psnr_v]
        curr_ssim = [ssim, ssim_y, ssim_u, ssim_v]
    else:
        assert args['src_type'] == 'png'
        rgb_rec = ycbcr2rgb(x_hat)
        rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        psnr = calc_psnr(rgb, rgb_rec)
        if args['calc_ssim']:
            msssim = calc_msssim_rgb(rgb, rgb_rec)
        else:
            msssim = 0.
        curr_psnr = [psnr]
        curr_ssim = [msssim]
    return curr_psnr, curr_ssim

def run_one_point_fast(optical_net, args):
    frame_num = args['frame_num']
    verbose = args['verbose']
    device = next(optical_net.parameters()).device

    flows = []
    inter_psnrs = []
    hist_diffs =  []
    start_time = time.time()
    src_reader = get_src_reader(args)
    pic_height = args['src_height']
    pic_width = args['src_width']
    padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)
    with torch.no_grad():
        for frame_idx in range(frame_num):
            frame_start_time = time.time()
            x_yuv, y, u, v, x_rgb, rgb = get_src_frame(args, src_reader, device)
            if frame_idx == 0:
                x_1 = replicate_pad(x_rgb, padding_b, padding_r)
                flow = 0
                warppsnr = 0
                inter_psnr = 0
                hist_diff =0
            if frame_idx > 0:
                x_2 = replicate_pad(x_rgb, padding_b, padding_r)
                flow = optical_net(x_1, x_2)
                warpframe = block_mc_func(x_1, flow)
                warpmse = F.mse_loss(warpframe, x_2)
                warppsnr = 10 * (torch.log(1 * 1 / warpmse) / np.log(10)).cpu().numpy()
                flow = flow.abs().mean().cpu().numpy()
                inter_mse = F.mse_loss(x_2, x_1)
                inter_psnr = 10 * (torch.log(1 * 1 / inter_mse) / np.log(10)).cpu().numpy()
                hist_diff = torch.abs(x_1.cpu().histc(bins=256) - x_2.cpu().histc(bins=256)).sum().numpy()
                x_1 = x_2
            hist_diffs.append(hist_diff)
            inter_psnrs.append(inter_psnr)
            flows.append(flow)
            frame_end_time = time.time()
            if verbose >= 2:
                print(f"frame {frame_idx}, {frame_end_time - frame_start_time:.3f} seconds, "
                      f"flow: {flow:.3f}, "
                      f"warppsnr: {warppsnr:.3f}, "
                      )

    src_reader.close()

    log_result = generate_log_json(frame_num - 1, flows,inter_psnrs, hist_diffs)
    return log_result



optical_net = None  # the model is initialized after each process is spawn, thus OK for multiprocess


def worker(args):
    global optical_net

    sub_dir_name = args['seq']
    bin_folder = os.path.join(args['stream_path'], args['ds_name'])
    assert args['write_stream'], ""
    create_folder(bin_folder, True)

    args['src_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['bin_folder'] = bin_folder
    args['curr_bin_path'] = os.path.join(bin_folder,
                                         f"{args['seq']}_q{args['qp_i']}.bin")
    args['curr_rec_path'] = args['curr_bin_path'].replace('.bin', '.yuv')
    args['curr_json_path'] = args['curr_bin_path'].replace('.bin', '.json')
    result = run_one_point_fast(optical_net, args)  

    result['ds_name'] = args['ds_name']
    result['seq'] = args['seq']
    result['rate_idx'] = args['rate_idx']
    result['qp_i'] = args['qp_i']
    result['qp_p'] = args['qp_p'] if 'qp_p' in args else args['qp_i']

    return result


def init_func(args, gpu_num):
    set_torch_env()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        if args.cuda_idx is not None:
            gpu_id = args.cuda_idx[gpu_id]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = "cuda:0"
    else:
        device = "cpu"

    global optical_net
    optical_net = ME_Spynet()
    optical_state = get_state_dict(args.model_path)
    optical_net.load_state_dict(optical_state)
    optical_net = optical_net.to(device)
    optical_net.eval()


def main():
    begin_time = time.time()

    args = parse_args()

    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None

    if args.cuda_idx is not None:
        cuda_device = ','.join([str(s) for s in args.cuda_idx])
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    gpu_num = 0
    if args.cuda:
        gpu_num = torch.cuda.device_count()

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num,
                                                                 initializer=init_func,
                                                                 initargs=(args, gpu_num))
    objs = []

    count_frames = 0
    count_sequences = 0

    rate_num = args.rate_num
    qp_i = []
    if args.qp_i is not None:
        assert len(args.qp_i) == rate_num
        qp_i = args.qp_i
    else:
        assert 2 <= rate_num <= DMC.get_qp_num()
        for i in np.linspace(0, DMC.get_qp_num() - 1, num=rate_num):
            qp_i.append(int(i+0.5))

    if not args.force_intra:
        if args.qp_p is not None:
            assert len(args.qp_p) == rate_num
            qp_p = args.qp_p
        else:
            qp_p = qp_i
    # qp_i = qp_p

    print(f"testing {rate_num} rates, using qp: ", end='')
    for q in qp_i:
        print(f"{q}, ", end='')
    print()

    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['qp_i'] = qp_i[rate_idx]
                if not args.force_intra:
                    cur_args['qp_p'] = qp_p[rate_idx]
                cur_args['force_intra'] = args.force_intra
                cur_args['reset_interval'] = args.reset_interval
                cur_args['seq'] = seq
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq]['width']
                cur_args['intra_period'] = config[ds_name]['sequences'][seq]['intra_period']
                if args.force_intra:
                    cur_args['intra_period'] = 1
                if args.force_intra_period > 0:
                    cur_args['intra_period'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['calc_ssim'] = args.calc_ssim
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['check_existing'] = args.check_existing
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose
                cur_args['verbose_json'] = args.verbose
                cur_args['run_fast'] = args.run_fast
                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(worker, cur_args)
                objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}

    for res in results:
        log_result[res['ds_name']][res['seq']][f"{res['rate_idx']:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
