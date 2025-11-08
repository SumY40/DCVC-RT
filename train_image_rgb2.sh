CUDA_VISIBLE_DEVICES=5 torchrun --nproc-per-node=1 \
                                --master_port=142 \
                                train_image_rgb_gan.py  \
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
                                --pretrained_weights \
                                -weight 1 \
                                -qn 64\  
