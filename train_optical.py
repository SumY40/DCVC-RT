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
from src.models.optical import ME_Spynet, block_mc_func
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import re
from torchvision import transforms
from src.utils.stream_helper import  get_state_dict
from datasets.dataset import  VimeoDataset
from src.utils.transforms import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
random.seed(0)
from torch.nn.parallel import DistributedDataParallel as DDP

print_step = 40
cal_step = 10
tot_epoch = 150
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
        



def train_one_epoch(epoch, global_step, optimizer, train_loader, frame_num):
    print ("epoch", epoch)
    optical_net.train()

    save_step = 5000
    save_name  = "optical_frame_" + "_" + args.train_mode + "_"

    bat_cnt = 0
    cal_cnt = 0
    sumpsnr = 0
    sumbpp = 0
    sumloss = 0
    summse = 0
    sumbpp_y = 0
    sumbpp_z = 0
    frame_count = 0
    pbar = tqdm.tqdm(train_loader, desc='epoch' + str(epoch), ncols=100,mininterval=10)
    for iteration, sample in enumerate(pbar):
        global_step += 1
        bat_cnt += 1
        total_loss = 0.0
        # for frame_idx in range(frame_num):

        im1 = sample[0].to(device)
        im2 = sample[1].to(device)

        flow = optical_net(im1, im2)

        warpframe = block_mc_func(im1, flow)

        mse = F.mse_loss(warpframe, im2)

        loss = mse

        loss.backward()
        nn.utils.clip_grad_norm_(optical_net.module.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()


        if global_step % cal_step == 0:
            cal_cnt += 1
            if mse > 0:
                psnr = 10 * (torch.log(1 * 1 / mse) / np.log(10)).cpu().detach().numpy()
            else:
                psnr = 100
            sumloss += loss.cpu().detach()
            summse += mse.cpu().detach()
            sumpsnr += psnr


        if (iteration % print_step)== 0 and bat_cnt > 1 :
            print("\n")
            if args.local_rank == 0:
                logging.info(
                    f'[{iteration}/{len(train_loader)}/{100. * iteration / len(train_loader):.2f}%] | '
                    f'Loss: {sumloss/cal_cnt:.5f} | '
                    f'warp psnr: {sumpsnr/cal_cnt:.5f} | '
                )

            bat_cnt = 0
            cal_cnt = 0
            sumpsnr = summse = sumloss = 0
        if global_step % save_step == 0:
            torch.save(optical_net.module.state_dict(), save_path + save_name + str(args.train_frame_num)+"_epoch_"+str(epoch)+"iter_"+str(global_step)+".pth")
            print("model saved!")

    return global_step



if __name__ == "__main__": 


    
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

    global optical_net

    
    optical_net = ME_Spynet()
    optical_net = optical_net.to(device)

    # if args.pretrained_weights:
    #     pretrained_dict = torch.load(args.pretrained_weights,map_location=torch.device('cpu'))
    #     model_dict = net.state_dict()
    #     filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}       
    #     model_dict.update(filtered_dict)
    #     net.load_state_dict(model_dict)
        # net.load_state_dict(torch.load(args.pretrained_weights))
    # 加载预训练权重
    if args.pretrained_weights:
        pretrained_dict = get_state_dict(args.pretrained_weights)
        model_dict = optical_net.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        # 加载过滤后的权重
        optical_net.load_state_dict(model_dict)
        print(f"加载了 {len(filtered_dict)} 个匹配的参数，总共有 {len(model_dict)} 个参数")
        match = re.search(r"(\d+)(?=\.pth$)", args.pretrained_weights)
        if match !=None:
            global_step = int(match.group(1))



    optical_net = nn.parallel.DistributedDataParallel(optical_net, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)

    
    base_lr = args.base_lr
    params = set()
    for n, p in optical_net.module.named_parameters():
        p.requires_grad_(True)
        params.add(p)
    optimizer = optim.AdamW(params, lr=base_lr)



    print("===start lr: ", base_lr, "===")
    save_path = "./ckpts/optical/"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    setup_logger("./train_log/optical_" + str(args.train_frame_num) + '.log')
    if args.local_rank == 0:
        logging.info(time.strftime('%Y%m%d_%H%M%S'))


    for epoch in range(0, tot_epoch):
        # adjust_learning_rate_rc(optimizer, epoch, start_lr=base_lr)
        if args.local_rank == 0:
            logging.info('====== Current epoch %s ======'%epoch)
            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_sampler.set_epoch(epoch) # shuffle
        # adjust_learning_rate_rc(optimizer, epoch, start_lr=base_lr)
        global_step = train_one_epoch(epoch, global_step, optimizer, train_loader, args.train_frame_num)
        # scheduler.step()
