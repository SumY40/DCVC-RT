import math
import re
import shutil

import torch
import numpy as np
import os, glob, argparse, time, random, logging
from tqdm import tqdm
from tensorboardX import SummaryWriter
from src.models.image_model import DMCI
from src.models.video_model import DMC

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.device_count())
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
import torchvision.transforms as transforms
from datasets.dataset import VimeoDataset
from torch.utils.data import DataLoader
from utils.loss_function import mse_yuv_loss, msssim_loss, mse_RGB_loss
from src.utils.common import get_state_dict

from src.utils.transforms import rgb2ycbcr, ycbcr2rgb


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default='/media/ivc-18/disk21/zjz/data/NVC_Dataset/training/vimeo_septuplet')

    parser.add_argument('--i_frame_model_path', type=str, default="./checkpoints/cvpr2025_image.pth.tar")
    # parser.add_argument('--i_frame_model_path', type=str, default="")
    parser.add_argument('--qp_num', type=int, default=64)
    parser.add_argument('-step', '--global_step', type=int, default=0)
    parser.add_argument('--train_frame_num', type=int, default=7)
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256),
                        help="Size of the patches to be cropped (default: %(default)s)", )
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Dataloaders worker per trainer')

    parser.add_argument("--mode", default='')
    parser.add_argument("--init_ckpt", default='')
    parser.add_argument('--pretrained_weights', type=str, default="",
                        help='load pre-trained weights for faster training; ignored if there are '
                             'existing checkpoints')
    parser.add_argument('--frozen_modules', nargs='+')
    parser.add_argument("--alpha", type=float, default=1.,
                        help="weights for distoration.")  # 100, 400, 1000, 2000, 4000
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")
    parser.add_argument("--fixed_rate", type=bool, default=True, help="train with fixed rate")
    parser.add_argument("--fixed_rate_num", type=int, default=63, help="fixed rate num in 0, 63")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=100)
    # parser.add_argument("--epoch_step", type=int, default=30)
    parser.add_argument("--check_time", type=float, default=3, help='frequency for recording state (min).')
    parser.add_argument("--prefix", type=str, default='tp',
                        help="prefix of checkpoints/logger, etc.")
    parser.add_argument('--mse_RGB', action='store_true', default=True, help='Use mse')
    parser.add_argument('--mse_YUV', action='store_true', default=False, help='Use mse')
    parser.add_argument('--msssim', action='store_true', default=False, help='Use msssim')
    parser.add_argument('--train_mode', type=str, default="allRD")

    args = parser.parse_args()

    return args


def get_cur_lamda(cur_qp, min_lamda=1, max_lamda=768, q_num=72):
    if isinstance(cur_qp, torch.Tensor):
        cur_qp = cur_qp.item()
    cur_lamda = np.exp(np.log(min_lamda) + (cur_qp / (q_num - 1)) * (np.log(max_lamda) - np.log(min_lamda)))
    return cur_lamda


class Trainer():
    def __init__(self, config, intra_model, model, args):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.writer = SummaryWriter(log_dir=config.logdir)
        self.intra_model = intra_model
        self.model = model
        self.logits = torch.tensor([1.0, 1.0]).to(device)
        self.logger.info(model)
        self.load_intra_model(self.config.init_intra_ckpt)  # load ckpt
        _ = self.load_inter_mdoel_dict(self.config.init_ckpt, record_global_step=True)  # load ckpt
        if args.mse_RGB:
            self.crit = mse_RGB_loss
        elif args.mse_YUV:
            self.crit = mse_yuv_loss
        self.epoch = 0
        self.record_set = {'loss': [], 'mse': [], 'msssim': [], 'psnr': [], 'bpp': [], 'bpp_y': [], 'bpp_z': []}

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def load_intra_model(self, init_ckpt):
        if init_ckpt == '':
            self.logger.info('Random initialization.')
            self.intra_model = self.intra_model.to(device)
        else:
            i_state_dict = get_state_dict(init_ckpt)
            self.intra_model.load_state_dict(i_state_dict)
            self.intra_model = self.intra_model.to(device)
            self.intra_model.eval()
            self.intra_model.update(None)
            self.logger.info('Load checkpoint from ' + init_ckpt)

    def load_inter_mdoel_dict(self, init_ckpt, record_global_step=False):
        if init_ckpt == '':
            self.logger.info('Random initialization.')
            self.model = self.model.to(device)
            pretrained_dict_keys = {}
        else:
            ckpt = torch.load(init_ckpt, map_location=device)
            checkpoint_state_dict = ckpt['model']
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            self.logger.info('Load checkpoint from ' + init_ckpt)
            print(f"Reload {len(pretrained_dict)} nums parameters，Total has {len(model_dict)} nums parameters")
            self.model.update(None)
            self.model = self.model.to(device)
            pretrained_dict_keys = [k.split('.')[0] for k in pretrained_dict.keys()]
            pretrained_dict_keys = np.unique(pretrained_dict_keys).tolist()
            self.logger.info('Load pretained modules:' + str(pretrained_dict_keys))
            # self.model.model.load_state_dict(ckpt['model'])
            if record_global_step:
                match = re.search(r"(\d+)(?=\.pth$)", init_ckpt)
                if match != None:
                    self.config.global_step = int(match.group(1))
                print(f"Global step: {self.config.global_step}")

        return pretrained_dict_keys

    def save_model(self, global_step):
        torch.save({'model': self.model.state_dict()},
                   os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth'))

        return

    def set_optimizer(self):
        self.logger.info('=' * 5 + ' set_optimizer' + '=' * 5)
        params_lr_list = []
        for module_name in self.model._modules.keys():
            if self.config.frozen_modules and module_name in self.config.frozen_modules:
                for v in self.model._modules[module_name].parameters(): v.requires_grad = False
                self.logger.info('frozen: ' + module_name)
            else:
                lr = self.config.lr
                params_lr_list.append({"params": self.model._modules[module_name].parameters(), 'lr': lr})
                self.logger.info('optimize: ' + module_name + '\tlr:' + str(lr))

        # 2. 单独添加直接定义的 nn.Parameter
        direct_params = []
        for name, param in self.model.named_parameters():
            # 检查参数是否属于任何子模块
            if '.' not in name:  # 直接定义的参数（如 q_encoder）
                direct_params.append(param)
                self.logger.info(f'optimize direct parameter: {name}, lr={self.config.lr}')

        if direct_params:
            params_lr_list.append({"params": direct_params, 'lr': self.config.lr})

        optimizer = torch.optim.AdamW(params_lr_list)

        return optimizer

    def clip_gradient(self, max_norm=None, clip_value=None, CHECK=False):
        # print('dbg: clip_gradient', [p.grad.data.abs().max().cpu().detach() for p in self.model.parameters()])
        if CHECK: max_grad_before = max(p.grad.data.abs().max() for p in self.model.parameters())
        # clip by norm
        if max_norm is not None: total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        if clip_value is not None: torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value)
        # # check
        if CHECK:
            total_norm_after = 0
            for p in self.model.parameters(): total_norm_after += p.grad.data.norm(2) ** 2
            total_norm_after = total_norm_after ** (1. / 2)
            max_grad_after = max(p.grad.data.abs().max() for p in self.model.parameters())
            print('total_norm:', total_norm.cpu().detach().numpy().round(4), ' -> ',
                  total_norm_after.cpu().detach().numpy().round(4), '\tthreshold:', max_norm)
            print('max_grad:', max_grad_before.cpu().detach().numpy().round(4), ' -> ',
                  max_grad_after.cpu().detach().numpy().round(4), '\tthreshold:', clip_value)

        return

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('=' * 10 + main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items():
            self.record_set[k] = np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items():
            self.logger.info(k + ': ' + str(np.round(v, 5).tolist()))
            if len(v.shape) > 1: self.logger.info(k + ': ' + str(np.round(v.sum(axis=-1), 5).tolist()))
        # tensorboard
        tag_scalar_dict = {}
        for k, v in self.record_set.items(): tag_scalar_dict[k] = np.sum(v)
        self.writer.add_scalars(main_tag=main_tag,
                                tag_scalar_dict=tag_scalar_dict,
                                global_step=global_step)
        # return zero
        for k in self.record_set.keys():
            self.record_set[k] = []

        return

    # @torch.no_grad()
    # def test(self, dataloader, main_tag='Test'):
    #     self.logger.info('Testing Files length:' + str(len(dataloader)))
    #     for _, (first_gt_coords, first_gt_feats, next_gt_coords, next_gt_feats,first_gpcc_coords, first_gpcc_feats, next_gpcc_coords, next_gpcc_feats) in enumerate(tqdm(dataloader)):
    #         # data
    #         ref = ME.SparseTensor(features=first_gpcc_feats.float(), coordinates=first_gpcc_coords, device=device)
    #         curr_gt = ME.SparseTensor(features=next_gt_feats.float(), coordinates=next_gt_coords, device=device)
    #         x = ME.SparseTensor(features=next_gpcc_feats.float(), coordinates=next_gpcc_coords, device=device)
    #         if x.shape[0] > self.config.max_num_points:
    #             self.logger.info('!!!!!!!!! GPU Memory !!!!!!!!! ' + str(x.shape[0]))
    #             continue
    #         # forward
    #         out_set = self.model(x, curr_gt, ref, training=False, k=self.config.k)
    #         if isinstance(out_set, list): out_set = out_set[-1]
    #         if not self.config.attr:
    #             bce = self.geo_crit(out_set['out_cls'], out_set['gt_coords'])
    #             loss = self.config.beta * bce
    #             print(bce)
    #         else:
    #             mse = self.crit(out_set['out'].F[:,:1], out_set['gt'].F[:,:1]).to(device)
    #             # mse = self.crit(out_set['out'].F[:,1:2], out_set['gt'].F[:,1:2]).to(device)
    #             # mse = self.crit(out_set['out'].F[:,2:], out_set['gt'].F[:,2:]).to(device)
    #             loss = self.config.alpha * mse
    #             print('enhanced:\t', mse * 10000)
    #         # record
    #         self.record_set['loss'].append(loss.item())
    #         if not self.config.attr:
    #             self.record_set['bce'].append(bce.item())
    #         else:
    #             self.record_set['mse'].append(mse.item())
    #         torch.cuda.empty_cache()  # empty cache.
    #     self.record(main_tag=main_tag, global_step=self.epoch)
    #
    #     return

    def train(self, dataloader, global_step):
        self.logger.info('=' * 40 + '\n' + 'Training Epoch: ' + str(self.epoch) + ', Global Step:' + str(global_step))
        quant_steps = list(range(self.config.qp_num))
        self.model.train()
        # optimizer
        self.optimizer = self.set_optimizer()
        self.logger.info('alpha:' + str(round(self.config.alpha, 2)) + '\tbeta:' + str(round(self.config.beta, 2)))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))
        start_time = time.time()
        # index_map = [0, 1, 0, 2, 0, 2, 0, 2]
        index_map = [0, 1, 0, 2, 0, 2, 2]

        # train_pattern = [0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]
        train_pattern = [1, 1, 1, 1, 1, 1, 1, 1]
        for batch_step, sample in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            global_step += 1
            total_loss = 0.0
            if self.config.fixed_rate:
                quant_step_list = [self.config.fixed_rate_num] * sample[0].shape[0]
            else:
                quant_step_list = random.choices(quant_steps, k=sample[0].shape[0])

            sum_mse, sum_warpmse, sum_bpp, sum_bpp_y, sum_bpp_z, sum_bpp_mv, sum_bpp_mv_y, sum_bpp_mv_z, sum_msssim, sum_mse_loss, sum_bpp_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0
            for frame_idx in range(self.config.train_frame_num):
                input_image = sample[frame_idx].to(device)
                if frame_idx == 0:
                    with torch.no_grad():
                        input_image = rgb2ycbcr(input_image)
                        i_result = self.intra_model(input_image, quant_step_list)
                        x_hat = ycbcr2rgb(i_result["x_hat"])
                        self.model.clear_dpb()
                        self.model.add_ref_frame(None, x_hat)
                else:
                    fa_idx = index_map[frame_idx % 8]
                    weight = train_pattern[frame_idx % 8]
                    qp_tensor = self.model.get_q_tensor(quant_step_list, device=input_image.device)
                    qp_tensor = self.model.shift_qp(qp_tensor, fa_idx)
                    lambda_list = [get_cur_lamda(quant_step) for quant_step in qp_tensor]
                    inter_result = self.model(input_image, qp_tensor)
                    recon_frame = inter_result["x_hat"].clamp(min=0, max=1)
                    msssim_perbatch = msssim_loss(recon_frame, input_image)
                    bpp_perbatch = inter_result["bpp"]
                    bpp_y = torch.mean(inter_result["bpp_y"])
                    bpp_z = torch.mean(inter_result["bpp_z"])
                    mse_perbatch = self.crit(recon_frame, input_image)
                    mse = torch.mean(mse_perbatch)
                    bpp = torch.mean(bpp_perbatch)
                    msssim = torch.mean(msssim_perbatch)
                    sum_bpp += bpp.item()
                    sum_bpp_y += bpp_y.item()
                    sum_bpp_z += bpp_z.item()
                    sum_mse += mse.item()
                    sum_msssim += msssim.item()
                    lambda_tensor = torch.tensor(lambda_list, device=device, dtype=input_image.dtype)
                    if self.config.train_mode == "allRD":
                        mse_loss = torch.mean(weight * lambda_tensor * mse_perbatch)
                        bpp_loss = torch.mean(bpp_perbatch)
                        loss = torch.mean(weight * lambda_tensor * mse_perbatch) + torch.mean(bpp_perbatch)
                    elif self.config.train_mode == "allD":
                        loss = torch.mean(weight * lambda_tensor * mse_perbatch)
                        # print('loss:', loss, ' mse_perbatch:', mse_perbatch, ' lambda:', lambda_tensor)
                    total_loss += loss
                    sum_mse_loss += mse_loss
                    sum_bpp_loss += bpp_loss

            mean_loss = total_loss / (self.config.train_frame_num - 1)
            sum_bpp /= (self.config.train_frame_num - 1)
            sum_bpp_y /= (self.config.train_frame_num - 1)
            sum_bpp_z /= (self.config.train_frame_num - 1)
            sum_mse /= (self.config.train_frame_num - 1)
            sum_msssim /= (self.config.train_frame_num - 1)
            sum_psnr = 10 * math.log10(1 / sum_mse)
            sum_mse_loss /= (self.config.train_frame_num - 1)
            sum_bpp_loss /= (self.config.train_frame_num - 1)

            if batch_step > 0:
                with torch.no_grad():
                    delta_log = torch.stack([
                        torch.log(previous_bpp) - torch.log(sum_bpp_loss),
                        torch.log(previous_mse) - torch.log(sum_mse_loss)
                    ]).cuda()
                    jacobian = torch.autograd.functional.jacobian(lambda xi: torch.softmax(xi, dim=0),
                                                                  self.logits).cuda()
                    delta_t = torch.matmul(jacobian.T, delta_log)
                    self.logits -= 0.001 * (delta_t + 0.025 * self.logits)
            # data
            # forward
            # mean_loss.backward()
            self.weight_distortion(sum_mse_loss, sum_bpp_loss, mean_loss)
            self.clip_gradient(max_norm=1.0)
            self.optimizer.step()
            previous_mse = sum_mse_loss
            previous_bpp = sum_bpp_loss
            # 检查梯度的，在optimizer.step()后
            # print("===grad===")
            # for name, parms in self.model.named_parameters():
            #     print('-->name:', name)
            #     # print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     if parms.grad != None:
            #         print('-->grad_value:', parms.grad.mean(), parms.grad.min(), parms.grad.max())
            #         print('-->grad_fn:', parms.grad_fn)
            #     else:
            #         print('-->grad_value: None')
            #     print("===")
            # record
            with torch.no_grad():
                self.record_set['loss'].append(mean_loss.item())
                self.record_set['mse'].append(sum_mse)
                self.record_set['msssim'].append(sum_msssim)
                self.record_set['psnr'].append(sum_psnr)
                self.record_set['bpp'].append(sum_bpp)
                self.record_set['bpp_y'].append(sum_bpp_y)
                self.record_set['bpp_z'].append(sum_bpp_z)
                if (time.time() - start_time) > self.config.check_time * 60:
                    self.record(main_tag='Train', global_step=global_step)
                    self.save_model(global_step=global_step)
                    start_time = time.time()
            torch.cuda.empty_cache()  # empty cache.
            torch.cuda.empty_cache()  # empty cache.
        with torch.no_grad():
            self.record(main_tag='Train', global_step=global_step)
        self.save_model(global_step=global_step)
        self.epoch += 1

        return global_step

    def weight_distortion(self, distortion, rate, loss):
        w = torch.softmax(self.logits, dim=0)
        wR, wD = w[0], w[1]
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # distortion.backward(retain_graph=True)

        grad_D = [p.grad.clone() for p in self.model.parameters()]
        grad_R = [p.grad.clone() for p in self.model.parameters()]
        log_grad_R = [g / rate.detach() for g in grad_R]
        log_grad_D = [g / distortion.detach() for g in grad_D]
        ct = 1.0 / (wR / rate.detach() + wD / distortion.detach())
        balanced_grad = [ct * (wR * lgr + wD * lgd) for lgr, lgd in zip(log_grad_R, log_grad_D)]
        i = 0
        for p in self.model.parameters():
            p.grad = balanced_grad[i]
            i += 1
        # for p, g in zip(self.model.parameters(), balanced_grad):
        #     p.data -= self.optimizer.param_groups[0]['lr'] * g


class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_intra_ckpt, init_ckpt, frozen_modules, alpha, beta, lr, check_time,
                 train_frame_num, train_mode, qp_num, global_step, fixed_rate, fixed_rate_num):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_intra_ckpt = init_intra_ckpt
        self.init_ckpt = init_ckpt
        self.frozen_modules = frozen_modules
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.check_time = check_time
        self.train_frame_num = train_frame_num
        self.train_mode = train_mode
        self.qp_num = qp_num
        self.global_step = global_step
        self.fixed_rate = fixed_rate
        self.fixed_rate_num = fixed_rate_num


if __name__ == '__main__':
    # log
    args = parse_args()

    # torch.manual_seed(0)
    # np.random.seed(seed=0)
    random.seed(0)
    intra_model = DMCI()
    inter_model = DMC()

    training_config = TrainingConfig(
        logdir=os.path.join('./logs', args.prefix),
        ckptdir=os.path.join('./ckpts', args.prefix),
        init_intra_ckpt=args.i_frame_model_path,
        init_ckpt=args.init_ckpt,
        frozen_modules=args.frozen_modules,
        alpha=args.alpha,
        beta=args.beta,
        lr=args.lr,
        check_time=args.check_time,
        train_frame_num=args.train_frame_num,
        train_mode=args.train_mode,
        qp_num=args.qp_num,
        global_step=args.global_step,
        fixed_rate=args.fixed_rate,
        fixed_rate_num=args.fixed_rate_num)
    # model
    shutil.copy('train_fm_zjz.py', os.path.join('./logs', args.prefix))
    gpus = list(range(torch.cuda.device_count()))
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(args.patch_size)]
    )

    # trainer
    trainer = Trainer(config=training_config, intra_model=intra_model, model=inter_model, args=args)
    # data
    train_dataset = VimeoDataset(
        args.dataset,
        rnd_interval=True,
        rnd_temp_order=True,
        split="train",
        transform=train_transforms,
        max_frames=args.train_frame_num,
    )
    # testing dataset
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=args.num_workers)
    # training
    global_step = 0
    for epoch in range(0, args.epoch):
        # training dataset

        # validate dataset
        # val_dataset = PC_Dyanamic_Dataset(filedirs[:round(len(filedirs) / 10)], have_label=True)
        # val_dataloader = make_dynamic_data_loader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)
        # update lr

        if epoch > 0 and epoch % 3 == 0 and epoch < 30: trainer.config.lr = max(trainer.config.lr * 0.8, args.lr_min)
        if epoch >= 15: trainer.config.lr = max(trainer.config.lr * 0.8, 5e-6)
        global_step = trainer.train(train_dataloader, global_step)
        # trainer.test(val_dataloader, 'Validate')
