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
from src.models.video_model_gan import DMC
from src.models.image_model import *
from pytorch_msssim import ms_ssim as MS_SSIM
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.rd_loss import get_relaxl1
import random
import re
from torchvision import transforms
from src.discriminator.discriminator import *
import lpips  # 引入 LPIPS 库
from src.utils.stream_helper import  get_state_dict
from my_datasets.dataset import VimeoDataset, BVIVimeoDatasetFromTxt
from src.utils.transforms import *
# from src.datasets.video_dataset import FastVideoFolder
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
random.seed(0)

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.loss_function import ALLYUVLoss, MSELossYUV2YUV
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
    parser.add_argument("-bpp", type=float, default = 3.5)
    parser.add_argument("-l1", type=float, default = 0.005)
    parser.add_argument("-lpips", type=float, default = 0.03)
    parser.add_argument("-artifact", type=float, default = 220)
    parser.add_argument("-g_weight", type=float, default = 0.01)
    parser.add_argument('-dl',  type=float, default=1e-4,
                        help='base learning rate')
    parser.add_argument("-d", "--train_dataset", type=str, default="/workspace/datasets/vimeo_septuplet/sequences/", help="Training dataset")
    parser.add_argument("-testd", "--test_dataset", type=str, default="/workspace/datasets/RL_video/RGB/", help="Testing dataset")
    # parser.add_argument("--load_checkpoint", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--pretrained_weights', type=str, default="",
                        help='load pre-trained weights for faster training; ignored if there are '
                             'existing checkpoints')
    parser.add_argument('--disc_weights', type=str, default="",
                        help='load pre-disc_weights for faster training; ignored if there are '
                             'existing checkpoints')
    parser.add_argument('--tensorboard_path', type=str, default="./0618/runs/",
                        help='path for tensorboard')
    parser.add_argument('--save_path', type=str, default="./0618/ckpts/",
                        help='path for saving ckpts')
    parser.add_argument('--save_imgs_path', type=str, default="./0618/imgs/",
                        help='path for saving ckpts')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--repeat', action='store_true', default=False, help='Use repeat')    
    parser.add_argument('--long', action='store_true', default=False, help='Use repeat')    
    parser.add_argument('-qn','--q_num', type=int, default=64)
    parser.add_argument('--fix_qp', type=int, default=0)
    parser.add_argument('--hira', type=int, default=1)
    parser.add_argument('--train_recon', action='store_true', default=False, help='Use repeat')    

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
        
def get_cur_lamda(cur_qp, min_lamda=1, max_lamda=768, q_num=72):
    cur_lamda = np.exp(np.log(min_lamda) + (cur_qp / (q_num - 1)) * (np.log(max_lamda) - np.log(min_lamda)))
    return cur_lamda
def msssim_loss(fake, real):
    assert fake.shape == real.shape
    return 1 - MS_SSIM(fake, real, data_range=1.0, size_average=False)   
bce = nn.BCEWithLogitsLoss()

def generator_loss_bce(discriminator, fake):
    fake_logits = discriminator(fake)
    labels = torch.ones_like(fake_logits)
    return bce(fake_logits, labels)
def discriminator_loss_bce(discriminator, real, fake):
    real_logits = discriminator(real)
    fake_logits = discriminator(fake.detach())

    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)

    loss_real = bce(real_logits, real_labels)
    loss_fake = bce(fake_logits, fake_labels)

    return 0.5 * (loss_real + loss_fake)

class hinge_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, for_real=True, for_G=False):
        # input: [B] 或 [B,1]
        if for_G:
            # Generator loss
            loss = -input.squeeze(1)  # [B] 保持 batch
        else:
            if for_real:
                loss = F.relu(1. - input)  # [B]
            else:
                loss = F.relu(1. + input)  # [B]
        return loss

def train_one_epoch(epoch, quant_steps, global_step, optimizer, train_loader, frame_num):
    print ("epoch", epoch)
    p_net.train()
    discriminator.train()
    i_net.eval()
    save_name  = "rt_frame_" + str(args.q_num) + "_" + args.train_mode + "_"
    if args.fix_qp == 63:
        save_name = save_name + "fix_qp63_"
    mseyuv_loss = MSELossYUV2YUV(gamma=1)
    index_map = [0, 1, 0, 2, 0, 2, 0 ,2]
    weight_map = [0.7, 1.0, 0.7, 0.9, 0.7, 0.9, 0.7, 0.9]

    q_num = args.q_num
    quant_steps = list(range(q_num))

    bce = torch.nn.BCEWithLogitsLoss(reduction='none')  # 不求平均，返回 shape 和输入一致
    loss_fn = ALLYUVLoss().to(device)
    def clip_gradient(optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
    if args.hira == 0:
        save_name  = save_name + "nohira_"
        pattern = [0, 0, 0, 0, 0, 0, 0, 0]
        index_map = [pattern[i % len(pattern)] for i in range(frame_num)]


    if args.train_recon:
        save_name = save_name + "recon_"
    if args.repeat:
        print("\n repeat first ")
        save_name  = save_name + "repeat_"
    gan_loss = hinge_loss()
    bat_cnt = 0
    cal_cnt = 0
    sumpsnr = 0
    sumbpp = 0
    sumloss = 0
    summse = 0
    sumbpp_y = 0
    sumbpp_z = 0
    sumdloss = 0
    sumgloss = 0
    frame_count = 0
    pbar = tqdm.tqdm(train_loader, desc='epoch' + str(epoch), ncols=100,mininterval=10)
    for iteration, sample in enumerate(pbar):
        global_step += 1
        bat_cnt += 1
        total_loss = 0.0
        total_d_loss = 0.0
        quant_step_list = random.choices(quant_steps, k=args.train_batch_size)
        if args.fix_qp == 63:
            quant_step_list = [63] *args.train_batch_size
        for frame_idx in range(frame_num):
            input_image = sample[frame_idx].to(device)
            input_image_yuv = rgb2ycbcr(input_image)
            if frame_idx == 0:
                # 将输入范围映射到输出范围
                with torch.no_grad():
                    # i_quant_step_list = [i //2 for i in quant_step_list]
                    i_result = i_net(input_image_yuv, quant_step_list)
                    x_hat_yuv = i_result["x_hat"]
                p_net.module.clear_dpb()
                p_net.module.add_ref_frame(None, x_hat_yuv)
                # mse_loss_fn = nn.MSELoss()
                # mse = torch.mean( mse_loss_fn(ref_image, x_hat))
                # psnr = 10 * (torch.log(1 * 1 / mse) / np.log(10)).cpu().detach().numpy()
                # print(psnr)
            else:
                fa_idx = index_map[frame_idx % 8]
                weight = weight_map[frame_idx % 8]
                if args.repeat:
                    if frame_idx == 1:
                        weight = 0.8
                        repeat_image_yuv = rgb2ycbcr(sample[0].to(device))
                        with torch.no_grad():
                            k = random.randint(1, 16)
                            for idx in range(k):
                                fa_idx = 0
                                qp_tensor = p_net.module.get_q_tensor(quant_step_list, device=input_image_yuv.device)
                                qp_tensor = p_net.module.shift_qp(qp_tensor, fa_idx)
                                p_result = p_net(repeat_image_yuv, qp_tensor)
                            p_net.module.clear_dpb()
                            p_net.module.add_ref_frame(None, p_result["x_hat"])

                # repeat first frame

                if global_step % 2 == 0:
                    # if frame_idx == 33:
                    #     p_net.module.clear_dpb()
                    #     p_net.module.add_ref_frame(None, p_result["x_hat"])
                    qp_tensor = p_net.module.get_q_tensor(quant_step_list, device=input_image_yuv.device)
                    qp_tensor = p_net.module.shift_qp(qp_tensor, fa_idx)
                    lambda_list = [get_cur_lamda(quant_step) for quant_step in quant_step_list]
                    p_result = p_net(input_image_yuv, qp_tensor)
                    loss_out = loss_fn(p_result, input_image_yuv)

                    # mse_loss_fn = nn.MSELoss()
                    # L1 loss
                    l1_loss = loss_out["l1_01"]
                    #ms-ssim
                    perceptual_loss = loss_out["lpipsnomean"]
                    ##g_loss
                    # if global_step % 2 == 0:
                    #     D_real_logits = discriminator(input_image)
                    #     D_gen_logits = discriminator(ycbcr2rgb(p_result["x_hat"]))
                    # else:
                    #     D_real_logits = discriminator(input_image.detach())
                    #     D_gen_logits = discriminator(ycbcr2rgb(p_result["x_hat"]).detach())
                    # D_real_logits = discriminator(input_image)
                    # D_gen_logits = discriminator()
                    # g_loss_perbatch = generator_loss_bce(discriminator, ycbcr2rgb(p_result["x_hat"]))
                    fake_pred = discriminator(ycbcr2rgb(p_result["x_hat"]))
                    g_loss_perbatch = gan_loss(fake_pred, for_G = True)

                    # g_loss_perbatch = gan_loss(D_real_logits, D_gen_logits, mode="generator_loss").mean(dim=[1, 2, 3])
                    # d_loss = gan_loss(D_real_logits, D_gen_logits, mode="discriminator_loss").mean()
                    ## mseloss
                    mse_perbatch = mseyuv_loss(input_image_yuv , p_result["x_hat"])
                    mse = torch.mean(mse_perbatch)
                    mse_rgb = (input_image -ycbcr2rgb( p_result["x_hat"])).pow(2)
                    # artifact
                    artifact_loss = loss_out["artifact_01"]
                    #bpp
                    bpp_perbatch = p_result["bpp"]
                    bpp = torch.mean(bpp_perbatch)
                    distortion = weight * (l1_loss * args.l1  + args.g_weight * g_loss_perbatch +  args.artifact * artifact_loss + args.lpips * perceptual_loss  )
                    if args.train_mode == "RD":
                        loss = torch.mean(  weight *  torch.tensor(lambda_list).cuda() * (l1_loss * args.l1  + args.g_weight * g_loss_perbatch +  args.artifact * artifact_loss + args.lpips * perceptual_loss   ) +  args.bpp * bpp_perbatch )
                        # loss = torch.mean(  torch.tensor(lambda_list).cuda() * (mse_perbatch + 0.006 * g_loss_perbatch   ) +  2.5 * bpp_perbatch )
                    elif args.train_mode == "D":
                        loss = torch.mean(  weight *  torch.tensor(lambda_list).cuda() * (mse_perbatch + 0.008 * g_loss_perbatch + 0.01 * perceptual_loss ) )
                    elif args.train_mode == "RDL1":
                        # loss = torch.mean(  torch.tensor(lambda_list).cuda() * (l1_loss  + 0.008 * g_loss_perbatch  + 0.01 * perceptual_loss  ) +  2.5 * bpp_perbatch )
                        loss = torch.mean(  weight *  torch.tensor(lambda_list).cuda() * (l1_loss * args.l1  + args.g_weight * g_loss_perbatch +  args.artifact * artifact_loss + args.lpips * perceptual_loss   ) +  args.bpp * bpp_perbatch )
                    # print("l1_loss", (l1_loss * args.l1 ).mean().cpu().detach().numpy())
                    # print('face_loss', (args.g_weight * g_loss_perbatch).mean().cpu().detach().numpy())
                    # print("g_loss", (args.g_weight * g_loss_perbatch).mean().cpu().detach().numpy())
                    # print("mse_loss", (0.8 * mse_perbatch.mean() + 0.2 * mse_rgb.mean()).mean().cpu().detach().numpy())
                    # print("bpp_loss", (args.bpp * bpp_perbatch).mean().cpu().detach().numpy())
                    # print("artifact_loss", (args.artifact * artifact_loss).mean().cpu().detach().numpy())
                    # print("perceptual_loss", (args.lpips * perceptual_loss).mean().cpu().detach().numpy())
                    # print("distortion", distortion.mean().cpu().detach().numpy())
                    total_loss += loss
                    frame_count += 1  # 更新帧数

                else:
                    # if frame_idx == 33:
                    #     p_net.module.clear_dpb()
                    #     p_net.module.add_ref_frame(None, p_result["x_hat"])
                    qp_tensor = p_net.module.get_q_tensor(quant_step_list, device=input_image_yuv.device)
                    qp_tensor = p_net.module.shift_qp(qp_tensor, fa_idx)
                    lambda_list = [get_cur_lamda(quant_step) for quant_step in qp_tensor.tolist()]
                    with torch.no_grad():
                        p_result = p_net(input_image_yuv, qp_tensor)
                    # D_real_logits = discriminator(input_image)
                    # D_gen_logits = discriminator(ycbcr2rgb(p_result["x_hat"]))
                    # # 相对平均输出（Relativistic average）
                    # D_ra_real = D_real_logits - D_gen_logits.mean()
                    # D_ra_fake = D_gen_logits - D_real_logits.mean()
                    # real_labels = torch.ones_like(D_real_logits)
                    # fake_labels = torch.zeros_like(D_gen_logits)

                    # d_loss_real = bce(D_ra_real, real_labels)
                    # d_loss_fake = bce(D_ra_fake, fake_labels)
                    # d_loss = (d_loss_real + d_loss_fake) / 2
                    # d_loss = d_loss.mean()
                    # d_loss = discriminator_loss_bce(discriminator, input_image, ycbcr2rgb(p_result["x_hat"]))        

                    D_fake = discriminator(ycbcr2rgb(p_result["x_hat"]))    # shape: (B, 1)
                    D_real = discriminator(input_image)    # shape: (B, 1)
                    loss_real = gan_loss(D_real, for_real=True)
                    loss_fake = gan_loss(D_fake, for_real=False)
                    d_loss = 0.5 * (loss_real + loss_fake)

                    total_d_loss += d_loss
                    frame_count += 1  # 更新帧数
        # print(bpp)
        # if g_loss_perbatch.mean() > 0.8:
        if global_step % 2 == 0:
            optimizer.zero_grad()
            if frame_count > 0:
                mean_loss = total_loss /frame_count
                mean_loss.backward()
                frame_count = 0
            nn.utils.clip_grad_norm_(p_net.module.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            disc_optimizer.zero_grad()
            if frame_count > 0:
                mean_d_loss = total_d_loss / frame_count
                mean_d_loss.backward()
                frame_count = 0
            nn.utils.clip_grad_norm_(discriminator.module.parameters(), max_norm=1.0)
            if mean_d_loss > 0.5:
                disc_optimizer.step()

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
            sumloss += loss.cpu().detach()
            sumgloss += g_loss_perbatch.mean().cpu().detach()
            sumdloss += d_loss.mean().cpu().detach()
            summse += mse.cpu().detach()
            sumpsnr += psnr
            sumbpp += bpp.cpu().detach()
            sumbpp_y += torch.mean(p_result["bpp_y"]).cpu().detach()
            sumbpp_z += torch.mean(p_result["bpp_z"]).cpu().detach()



        if (iteration % print_step)== 0 and bat_cnt > 1 :
            print("\n")
            if args.local_rank == 0:
                logging.info(
                    f'[{iteration}/{len(train_loader)}/{100. * iteration / len(train_loader):.2f}%] | '
                    f'Loss: {sumloss/cal_cnt:.5f} | '
                    f'g loss: {sumgloss/cal_cnt:.5f} | '
                    f'd loss: {sumdloss/cal_cnt:.5f} | '
                    f'bpp: { sumbpp/cal_cnt:.5f} | '
                    f'bpp_y: { sumbpp_y/cal_cnt:.5f} | '
                    f'bpp_z: { sumbpp_z/cal_cnt:.5f} | '
                    f'psnr: {sumpsnr/cal_cnt:.5f} | '
                )

            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumbpp_y = sumbpp_z = sumpsnr = sumdloss= summse = sumloss= sumgloss = 0
        if global_step % 1000 == 0:
            torch.save(p_net.module.state_dict(), save_path + save_name+str(args.train_frame_num)+"_epoch_"+str(epoch)+"iter_"+str(global_step)+".pth")
            # torch.save(discriminator.module.state_dict(),  "./disc/yuv_input/" + "disc_epoch_"+str(epoch)+"iter_"+str(global_step)+".pth")
            print("model saved!")

    return global_step



if __name__ == "__main__": 


    
    quant_steps = list(range(64))
    max_lamda = 768
    min_lamda = 1
    parser = argparse.ArgumentParser(description = 'Train')
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
    global dis
    
    p_net = DMC()
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

    p_state_dict = get_state_dict(args.pretrained_weights)
    p_net.load_state_dict(p_state_dict) 

    i_net =DMCI()
    i_net = i_net.to(device)
    i_state_dict = get_state_dict(args.i_frame_model_path)
    i_net.load_state_dict(i_state_dict)    

    discriminator = CLIPWithMLPDiscriminator()
    discriminator = discriminator.to(device) 

    if args.disc_weights:
        pretrained_dict = get_state_dict(args.disc_weights)
        model_dict = discriminator.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        # 加载过滤后的权重
        discriminator.load_state_dict(model_dict)
        print(f"加载了 {len(filtered_dict)} 个匹配的参数，总共有 {len(model_dict)} 个参数")



    p_net = nn.parallel.DistributedDataParallel(p_net, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)
    i_net = nn.parallel.DistributedDataParallel(i_net, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)    




    params = set()
    for n, p in p_net.module.named_parameters():
        p.requires_grad_(True)
        params.add(p)
    base_lr = args.base_lr
    optimizer = optim.AdamW(params, betas=(0.5, 0.9),lr=base_lr)
    if args.train_recon:
        params = set()
        for n,p in p_net.module.named_parameters():
            if "recon_generation_net" in n :
                p.requires_grad_(True)
                params.add(p)


    print("✅ 参与训练的判别器参数名称列表：")
    for name, param in discriminator.named_parameters():
        if param.requires_grad:
            print(name)
    disc_optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=args.base_lr,
        betas=(0.5, 0.9) # 可调，默认值建议加上
    )

    print("===start lr: ", base_lr, "===")
    save_path = "./ckpts/yuv_input/gan/clip/30k/"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    setup_logger("./train_log/mse_" + str(args.train_frame_num) + '.log')
    if args.local_rank == 0:
        logging.info(time.strftime('%Y%m%d_%H%M%S'))
    # p_net.module.update(None)
    # i_net.module.update(None)



    for epoch in range(0, tot_epoch):
        # adjust_learning_rate_rc(optimizer, epoch, start_lr=base_lr)
        if args.local_rank == 0:
            logging.info('====== Current epoch %s ======'%epoch)
            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_sampler.set_epoch(epoch) # shuffle
        # is_inter, is_residue, is_lambda_map = adjust_train_strategy(args.mode)
        # adjust_learning_rate_rc(optimizer, epoch, start_lr=base_lr)
        global_step = train_one_epoch(epoch, quant_steps, global_step, optimizer, train_loader, args.train_frame_num)
