import glob
import logging
import os
import argparse
import time
import sys
import torch
import numpy as np
import tqdm
# from src.models.video_model_realtime import *
from src.models.video_model import DMC
from src.models.image_model import *
from pytorch_msssim import ms_ssim as MS_SSIM
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data
import random
import re
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image
from src.utils.stream_helper import  get_state_dict
from datasets.dataset import VimeoDataset, MyLongDataset,  BVIVimeoDatasetFromTxt
# from src.datasets.video_dataset import FastVideoFolder
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
random.seed(200)
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.loss_function import msssim_loss, mse_RGB_loss, MSELossYUV2YUV
from src.utils.transforms import ycbcr2rgb, rgb2ycbcr
print_step = 40
cal_step = 10
tot_epoch = 100
tot_step = 2000000
global_step = 0
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train DCVC-FM")
    parser.add_argument('--i_frame_model_name', type=str, default="psnr")
    parser.add_argument('--i_frame_model_path', type=str, default="./checkpoints/cvpr2025_image.pth.tar")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4, help="Test batch size (default: %(default)s)",)
    parser.add_argument('--train_frame_num', type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256), help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Dataloaders worker per trainer')
    parser.add_argument('--train_mode', type=str, default="all")

    parser.add_argument('-l', '--base_lr', type=float, default=5e-6,
                        help='base learning rate')
    parser.add_argument("-d", "--train_dataset", type=str, default="/workspace/datasets/vimeo_septuplet/sequences/", help="Training dataset")
    parser.add_argument("-testd", "--test_dataset", type=str, default="/workspace/datasets/RL_video/RGB/", help="Testing dataset")
    # parser.add_argument("--load_checkpoint", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--pretrained_weights', type=str, default="",
                        help='load pre-trained weights for faster training; ignored if there are '
                             'existing checkpoints')
    parser.add_argument('--tensorboard_path', type=str, default="./0618/runs/",
                        help='path for tensorboard')
    parser.add_argument('--save_path', type=str, default="./ckpts/yuv_input/msssim/",
                        help='path for saving ckpts')
    parser.add_argument('--save_imgs_path', type=str, default="./0618/imgs/",
                        help='path for saving ckpts')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--repeat', action='store_true', default=False, help='Use repeat')    
    parser.add_argument('--long', action='store_true', default=False, help='Use repeat')    
    parser.add_argument('-qn','--q_num', type=int, default=64)
    parser.add_argument('-gamma', type=int, default=1)
    parser.add_argument('--fix_qp', type=int, default=0)
    parser.add_argument('--hira', type=int, default=1)
    parser.add_argument('--noise', type=int, default=0)
    args = parser.parse_args(argv)
    return args
def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir,mode='a', encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)
        



def adjust_learning_rate_rc(optimizer, epoch, start_lr=1e-4):
    """Sets the learning rate to the initial LR decayed by 5 every 10 epochs"""
    lr = start_lr * (0.65 ** epoch)
    if lr < 1e-6:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def get_cur_lamda(cur_qp, min_lamda=1, max_lamda=768, q_num=64):
    cur_lamda = np.exp(np.log(min_lamda) + (cur_qp / (q_num - 1)) * (np.log(max_lamda) - np.log(min_lamda)))
    return cur_lamda
def update_modules(model, force=True):
    for name, module in model.named_modules():
        if hasattr(module, 'update') and callable(getattr(module, 'update')):
            print(f"Module '{name}' has an update method.")
            module.update(force=force)
def msssim_loss(fake, real):
    assert fake.shape == real.shape
    return 1 - MS_SSIM(fake, real, data_range=1.0, size_average=False)   
def train_one_epoch(epoch, quant_steps, global_step, optimizer, train_loader, frame_num):
    print ("epoch", epoch)
    p_net.train()
    i_net.eval()

    if frame_num > 7:
        save_step = 1000
    else:
        save_step = 5000
    save_name  = "rt_frame_" + str(args.q_num) + "_" + args.train_mode + "_"
    if args.fix_qp == 60:
        save_name = save_name + "fix_qp60_"
    if args.noise:
        save_name = save_name + "noise_"
    mseyuv_loss = MSELossYUV2YUV(gamma=args.gamma)
    index_map = [0, 1, 0, 2, 0, 2, 0 ,2]
    weight_map = [0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]
    q_num = args.q_num
    quant_steps = list(range(q_num))

    if args.repeat:
        print("\n repeat first ")
        save_name  = save_name + "repeat_"

    if args.hira == 0:
        save_name  = save_name + "nohira_"
        index_map = [0, 0, 0, 0, 0, 0, 0, 0]
    def clip_gradient(optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
    bat_cnt = 0
    cal_cnt = 0
    sumpsnr = 0
    sumbpp = 0
    sumloss = 0
    summse = 0
    sumbpp_y = 0
    sumbpp_z = 0
    summse_rgb = 0
    sumpsnr_rgb = 0
    pbar = tqdm.tqdm(train_loader, desc='epoch' + str(epoch), ncols=100,mininterval=10)
    for iteration, sample in enumerate(pbar):
        global_step += 1
        bat_cnt += 1
        total_loss = 0.0
        quant_step_list = random.choices(quant_steps, k=args.train_batch_size)
        if args.fix_qp == 60:
            quant_step_list = [60] *args.train_batch_size
        for frame_idx in range(frame_num):
            input_image_rgb = sample[frame_idx].to(device)
            input_image_yuv = rgb2ycbcr(input_image_rgb)
  

            if frame_idx == 0:
                # 将输入范围映射到输出范围
                with torch.no_grad():
                    # i_quant_step_list = [i //2 for i in quant_step_list]
                    qp_tensor = p_net.module.get_q_tensor(quant_step_list, device=input_image_yuv.device)
                    i_result = i_net(input_image_yuv, quant_step_list)
                    x_hat_yuv = i_result["x_hat"]
                p_net.module.clear_dpb()
                p_net.module.add_ref_frame(None, x_hat_yuv)
                # 使用 MSELoss 计算均方误差
                # mse_loss_fn = nn.MSELoss()
                # mse = torch.mean( mse_loss_fn(input_image_yuv, x_hat_yuv))
                # psnr = 10 * (torch.log(1 * 1 / mse) / np.log(10)).cpu().detach().numpy()
                # print(psnr)
            else:
                if args.repeat:
                    if frame_idx == 1:
                        repeat_image_yuv = rgb2ycbcr(sample[0].to(device))
                        with torch.no_grad():
                            k = random.randint(1, 16)
                            for idx in range(k):
                                fa_idx = idx % 3
                                qp_tensor = p_net.module.get_q_tensor(quant_step_list, device=input_image_yuv.device)
                                qp_tensor = p_net.module.shift_qp(qp_tensor, fa_idx)
                                p_result = p_net(repeat_image_yuv, qp_tensor)
                            p_net.module.clear_dpb()
                            p_net.module.add_ref_frame(None, p_result["x_hat"])

                # if frame_idx % 64 == 1 and frame_idx !=1  :
                #     p_net.module.dpb[0].frame = p_result["x_hat"].clamp(0, 1)
                #     p_net.module.reset_ref_feature()

                fa_idx = index_map[frame_idx % 8]
                weight = weight_map[frame_idx % 8]
                if frame_idx == 1:
                    weight = 2.0
                # elif frame_idx == frame_num - 1:
                #     weight = 1.2 
                #     fa_idx = 1
                qp_tensor = p_net.module.get_q_tensor(quant_step_list, device=input_image_yuv.device)
                qp_tensor = p_net.module.shift_qp(qp_tensor, fa_idx)
                p_result = p_net(input_image_yuv, qp_tensor)
                lambda_list = [get_cur_lamda(quant_step) for quant_step in qp_tensor.tolist()]
                # lambda_list = [get_cur_lamda(quant_step) for quant_step in quant_step_list]
                msssim_yuv = msssim_loss(p_result["x_hat"], input_image_yuv)
                msssim_rgb = msssim_loss(ycbcr2rgb(p_result["x_hat"]), input_image_rgb)
                msssim = 0.8 * msssim_yuv + 0.2 * msssim_rgb
                bpp_perbatch = p_result["bpp"]
                mseyuv_perbatch = mseyuv_loss(p_result["x_hat"], input_image_yuv)
                msergb_perbatch = torch.mean((ycbcr2rgb(p_result["x_hat"]) - input_image_rgb) ** 2,dim=(1,2,3))
                mse = torch.mean(mseyuv_perbatch)
                msergb = torch.mean(msergb_perbatch)
                bpp = torch.mean(bpp_perbatch)
                if args.train_mode == "RD":
                    loss = torch.mean(  torch.tensor(lambda_list).cuda() * ( msssim / 64) + bpp_perbatch )
                elif args.train_mode == "D":
                    loss = torch.mean( torch.tensor(lambda_list).cuda() * ( msssim) )
                total_loss += loss

                # if frame_idx % 33 == 0:
                #     mean_loss = total_loss / 8
                #     mean_loss.backward()  
                #     total_loss = 0.0    
                #     p_net.module.dpb[0].feature = p_net.module.dpb[0].feature.detach()
        mean_loss = total_loss / (frame_num - 1)
        mean_loss.backward()
        clip_gradient(optimizer, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        # for name, param in p_net.module.named_parameters():
        #     if param.grad is not None:
        #         print(f"参数: {name}, 梯度范数: {torch.norm(param.grad).item()}") #打印每个参数的梯度范数
        #         # 如果你想打印完整的梯度值，可以使用：
        #         # print(f"参数: {name}, 梯度: {param.grad}")
        if global_step % cal_step == 0:
            cal_cnt += 1
            if mse > 0:
                psnr = 10 * (torch.log(1 * 1 / mse) / np.log(10)).cpu().detach().numpy()
            else:
                psnr = 100
            if msergb > 0:
                psnrrgb = 10 * (torch.log(1 * 1 / msergb) / np.log(10)).cpu().detach().numpy()
            else:
                psnrrgb = 100
            sumloss += mean_loss.cpu().detach()
            summse += mse.cpu().detach()
            summse_rgb += msergb.cpu().detach()
            sumpsnr += psnr
            sumpsnr_rgb += psnrrgb
            sumbpp += bpp.cpu().detach()
            sumbpp_y += torch.mean(p_result["bpp_y"]).cpu().detach()
            sumbpp_z += torch.mean(p_result["bpp_z"]).cpu().detach()


        if (iteration % print_step)== 0 and bat_cnt > 1 :
            print("\n")
            if args.local_rank == 0:
                logging.info(
                    f'[{iteration}/{len(train_loader)}/{100. * iteration / len(train_loader):.2f}%] | '
                    f'Loss: {sumloss/cal_cnt:.5f} | '
                    f'bpp: { sumbpp/cal_cnt:.5f} | '
                    f'bpp_y: { sumbpp_y/cal_cnt:.5f} | '
                    f'bpp_z: { sumbpp_z/cal_cnt:.5f} | '
                    f'psnr: {sumpsnr/cal_cnt:.5f} | '
                    f'psnrRGB: {sumpsnr_rgb/cal_cnt:.5f} | '
                )

            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumbpp_y = sumbpp_z = sumpsnr = summse_rgb= summse = sumloss= sumpsnr_rgb =sum_codebookloss= 0
        if global_step % save_step == 0:
            torch.save(p_net.module.state_dict(), save_path + save_name+str(args.train_frame_num)+"_epoch_"+str(epoch)+"iter_"+str(global_step)+".pth")
            print("model saved!")

    return global_step



if __name__ == "__main__": 


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gpus = [0, 1, 2, 3]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    
    quant_steps = list(range(64))
    max_lamda = 768
    min_lamda = 1
    supervision_epochs = 5
    parser = argparse.ArgumentParser(description = 'Train')
    # parser.add_argument('--mode',default=0, type=int, help ='the train mode')
    # parser.add_argument('--batch_nums',default=2, type=int, help ='the batch numbers')
    # args = parser.parse_args()
    args = parse_args(sys.argv[1:])    

    setup_logger("./train_log/msssim_" + str(args.train_frame_num) + '.log')
    print("===start lr: ", args.base_lr, "===")

    # 初始化分布式环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')



    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(args.patch_size)]
    )
    
    if args.train_frame_num > 7:
        train_dataset = BVIVimeoDatasetFromTxt(patch_size=args.patch_size,  max_frames=args.train_frame_num)
        # train_dataset = MyLongDataset(frame_num=args.train_frame_num)    
    else:
        train_dataset = VimeoDataset(rnd_temp_order=True ,max_frames=args.train_frame_num)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8, pin_memory=(device == "cuda"))

    global i_net
    global p_net

    
    p_net = DMC()
    if args.noise:
        p_net = DMC_noise()
    p_net = p_net.to(device)


    # 加载预训练权重
    if args.pretrained_weights:
        pretrained_dict = get_state_dict(args.pretrained_weights)
        model_dict = p_net.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        # 加载过滤后的权重
        p_net.load_state_dict(model_dict)
        print(f"加载了 {len(filtered_dict)} 个匹配的参数，总共有 {len(model_dict)} 个参数")
        match = re.search(r"(\d+)(?=\.pth$)", args.pretrained_weights)
        if match !=None:
            global_step = int(match.group(1))

    # # 加载网络
    # pretrained_dict = torch.load('./ckpts/yuv_input/rt_frame_64_D_3_epoch_7iter_605000.pth')
    # model_dict = p_net.state_dict()
    # filtered_dict = {k: v for k, v in pretrained_dict.items() 
    #                 if "bit_estimator_z" in k  }
    # for k in filtered_dict.keys():
    #     print(k)
    # model_dict.update(filtered_dict)
    # p_net.load_state_dict(model_dict)
    # print(f"加载了 {len(filtered_dict)} 个匹配的光流参数，总共有 {len(model_dict)} 个参数")


    i_net =DMCI()
    i_net = i_net.to(device)
    i_state_dict = get_state_dict(args.i_frame_model_path)
    i_net.load_state_dict(i_state_dict)    


    num_gpus = torch.cuda.device_count()
    print('use {} gpus!'.format(num_gpus))
    p_net = nn.parallel.DistributedDataParallel(p_net, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)
    i_net = nn.parallel.DistributedDataParallel(i_net, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)
        

    # params = set(p for n, p in p_net.named_parameters())
    params = set()
    for n, p in p_net.named_parameters():
        # if "bit_estimator_z" in n:
        # if n.startswith(("bit_estimator_z_mv", "mv_encoder", "mv_hyper_prior_encoder", "mv_hyper_prior_decoder", "mv_decoder")) == True:
        p.requires_grad_(True)
        params.add(p)
        # else:
        #     p.requires_grad_(False)
    base_lr = args.base_lr
    print("===start lr: ", base_lr, "===")
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    optimizer = optim.AdamW(params, lr=base_lr)

    setup_logger("./train_log/mseyuv_" + str(args.train_frame_num) + '.log')
    if args.local_rank == 0:
        logging.info(time.strftime('%Y%m%d_%H%M%S'))
    p_net.module.update(None)
    i_net.module.update(None)



    for epoch in range(0, tot_epoch):
        if args.local_rank == 0:
            logging.info('====== Current epoch %s ======'%epoch)
            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_sampler.set_epoch(epoch) # shuffle
        # is_inter, is_residue, is_lambda_map = adjust_train_strategy(args.mode)
        # if args.train_mode == "RD":
            # adjust_learning_rate_rc(optimizer, epoch, start_lr=base_lr)
        global_step = train_one_epoch(epoch, quant_steps, global_step, optimizer, train_loader, args.train_frame_num)
