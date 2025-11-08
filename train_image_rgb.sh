
# CUDA_VISIBLE_DEVICES=3,5  python -m torch.distributed.launch --nproc_per_node=2 \
#                                                             --master_port=655 \
#                                                             train_image_rgb.py  \
#                                                             --train_batch_size 16 \
#                                                             --train_mode D \
#                                                             -n 8 \
#                                                             -l 1e-4\
#                                                             -weight 8 \
#                                                             --patch_size 256 256 \



CUDA_VISIBLE_DEVICES=1,3,5  python -m torch.distributed.launch --nproc_per_node=3 \
                                                            --master_port=655 \
                                                            train_image_rgb.py  \
                                                            --train_batch_size 16 \
                                                            --train_mode RD \
                                                            -n 8 \
                                                            -l 1e-4\
                                                            -weight 16 \
                                                            --patch_size 256 256 \
                                                            --pretrained_weights /workspace/yx/DCVC/RT/ckpts/image/rgb_input/rt_intra_64_D__epoch_35iter_130000.pth 
