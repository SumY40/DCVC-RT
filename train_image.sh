
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


CUDA_VISIBLE_DEVICES=5  python -m torch.distributed.launch --nproc_per_node=1 \
                                                            --master_port=655 \
                                                            train_image_yuv.py  \
                                                            --train_batch_size 16 \
                                                            --train_mode RD \
                                                            -n 8 \
                                                            -l 1e-4\
                                                            --pretrained_weights /workspace/yx/DCVC/RT/ckpts/image/yuv_input/rt_intra_64_D__epoch_32iter_240000.pth \
                                                            -weight 16 \
                                                            --patch_size 256 256 \
