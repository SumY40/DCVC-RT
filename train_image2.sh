
# CUDA_VISIBLE_DEVICES=4  python -m torch.distributed.launch --nproc_per_node=1 \
#                                                             --master_port=655 \
#                                                             train_image_yuv.py  \
#                                                             --train_batch_size 16 \
#                                                             --train_mode D \
#                                                             -n 8 \
#                                                             -l 1e-4\                                        
#                                                             -weight 16 \
#                                                             --pretrained_weights /workspace/yx/DCVC/RT/ckpts/image/yuv_input/rt_intra_64_D__epoch_7iter_52000.pth \
#                                                             --patch_size 256 256 \
#                                                             --fix_qp 60 \


# CUDA_VISIBLE_DEVICES=4  python -m torch.distributed.launch --nproc_per_node=1 \
#                                                             --master_port=655 \
#                                                             train_image_yuv.py  \
#                                                             --train_batch_size 16 \
#                                                             --train_mode RD \
#                                                             -n 8 \
#                                                             -l 1e-6\
#                                                             --pretrained_weights /workspace/yx/DCVC/RT/ckpts/image/yuv_input/rt_intra_64_D__epoch_32iter_240000.pth \
#                                                             -weight 16 \
#                                                             --patch_size 256 256 \
# #                                                             
# CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 \
#                                                             --master_port=655 \
#                                                             train_image_rgb.py  \
#                                                             --train_batch_size 16 \
#                                                             --train_mode RD \
#                                                             -n 8 \
#                                                             -l 1e-6\
#                                                             --pretrained_weights /workspace/lxc/DCVC/RT/ckpts/image/rgb_input/rt_intra_64_D__epoch_22iter_170000.pth \
#                                                             -weight 16 \
#                                                             --patch_size 256 256 \
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 \
#                                                             --master_port=142 \
#                                                             train_image_yuv_gan.py  \
#                                                             --train_batch_size 32\
#                                                             --train_frame_num 1 \
#                                                             --train_mode RD\
#                                                             -n 8 \
#                                                             -lpips 0.006578 \
#                                                             -l1 1 \
#                                                             -l2 1 \
#                                                             -artifact 0.1 \
#                                                             -bpp 1 \
#                                                             -g_weight 0.0032\
#                                                             -l 5e-6\
#                                                             -dl 5e-6\
#                                                             --pretrained_weights ./checkpoints/cvpr2025_image.pth.tar\
#                                                             -weight 16 \
#                                                             -qn 32\

# CUDA_VISIBLE_DEVICES=2 torchrun --nproc-per-node=1 \
#                                 --master_port=142 \
#                                 train_image_yuv_gan.py  \
#                                 --train_batch_size 16\
#                                 --train_frame_num 1 \
#                                 --train_mode RD\
#                                 -n 8 \
#                                 -lpips 0.15 \
#                                 -l1 0.01 \
#                                 -l2 1 \
#                                 -artifact 1000 \
#                                 -bpp 1.0\
#                                 -g_weight 0.10 \
#                                 -l 5e-6\
#                                 -dl 5e-6\
#                                 --pretrained_weights /workspace/yx/DCVC/RT/ckpts/image/yuv_input/rt_intra_64_RD_lr1e-05__epoch_84iter_622000.pth\
#                                 -weight 1 \
#                                 -qn 32\
                                                    
#开始
# CUDA_VISIBLE_DEVICES=5  python -m torch.distributed.launch --nproc_per_node=1 \
#                                                             --master_port=29600 \
#                                                             train_image_yuv.py  \
#                                                             --train_batch_size 16 \
#                                                             --train_mode RD \
#                                                             -n 8 \
#                                                             -l 1e-5\
#                                                             --pretrained_weights /workspace/yx/DCVC/RT/ckpts/image/yuv_input/rt_intra_64_RD__epoch_12iter_92000.pth \
#                                                             -weight 16 \
#                                                             --patch_size 256 256 \


CUDA_VISIBLE_DEVICES=5 torchrun --nproc-per-node=1 \
                                --master_port=142 \
                                train_image_yuv_gan.py  \
                                --train_batch_size 16\
                                --train_frame_num 1 \
                                --train_mode RD\
                                -n 8 \
                                -lpips 0.15 \
                                -l1 0.01 \
                                -l2 1 \
                                -artifact 1000 \
                                -bpp 1.0\
                                -g_weight 0.10 \
                                -l 5e-6\
                                -dl 5e-6\
                                --pretrained_weights /workspace/yx/DCVC/RT/ckpts/image/yuv_input/rt_intra_64_RD_lr1e-05__epoch_84iter_622000.pth\
                                -weight 1 \
                                -qn 64\  