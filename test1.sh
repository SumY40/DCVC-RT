
# python compute_opticalflow.py   \
#              --model_path ./ckpts/optical/optical_frame__all_2_epoch_74iter_75000.pth\
#              --test_config ./config_huawei_all.json \
#              --save_decoded_frame True \
#              --cuda 1 -w 4\
#              --write_stream 1 \
#              --output_path out_log/output_optical.json \
#              --cuda_idx 4 2\
#              --reset_interval 32 \
#              --check_existing 0 \
#              --rate_num 1\
#              --qp_p 0 \
#              --qp_i 0   \
#              --force_intra_period 1\
#              --run_fast True \
#              --force_frame_num 96 \

# python test_video.py --model_path_i ./checkpoints/cvpr2025_image.pth.tar  \
#              --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
#              --test_config ./config_uvg_96_yuv.json \
#              --save_decoded_frame True \
#              --cuda 1 -w 4\
#              --write_stream 1 \
#              --output_path out_log/output_myrt_yuv.json \
#              --cuda_idx  6 7\
#              --reset_interval 32 \
#              --check_existing 0 \
#              --rate_num 6\
#              --force_intra_period 1\
#              --force_frame_num 96 \
#              --run_fast True \


  
# python test_video_allrgb.py --model_path_i  ./ckpts/images/yuv_input/gan/rt_intra_32_RD_1_epoch_13iter_50000.pth \
#              --model_path_p ./ckpts/rgb_input/rt_frame_64_RD_fix_qp60_7_lr1e-5.pth\
#              --test_config ./config_uvg_96_rgb_709.json \
#              --save_decoded_frame True \
#              --cuda 1 -w 4\
#              --write_stream 1 \
#              --output_path ./out_log/output_myrt_rgb.json \
#              --cuda_idx 2\
#              --reset_interval 32 \
#              --check_existing 0 \
#              --rate_num 2\
#              --qp_p 0 25 \
#              --qp_i 0 25 \
#              --force_frame_num 96 \
#              --force_intra_period 1\
#              --run_fast True \


# python test_video_gan.py --model_path_i ./ckpts/images/yuv_input/gan/tempbest_image.pth  \
# python test_video_gan.py --model_path_i ./ckpts/gan_all/intra_33_epoch_0iter_1000.pth  \
# python test_video_lpips.py --model_path_i ./checkpoints/cvpr2025_image.pth.tar  \
#              --model_path_p ./ckpts/yuv_input/gan/clip/rt_frame_32_RDL1_repeat_64_epoch_5iter_74000.pth\
#              --test_config ./config_yuv.json \
#              --save_decoded_frame True \
#              --cuda 1 -w 1\
#              --write_stream 1 \
#              --output_path output.json \
#              --cuda_idx 2\
#              --reset_interval 32 \
#              --check_existing 0 \
#              --rate_num 1\
#              --qp_p 30 \
#              --qp_i  30  \
#              --force_intra_period 11111\
#              --calc_ssim True \
#              --run_fast True \

# python test_video_gan.py --model_path_i ./ckpts/images/yuv_input/gan/rt_intra_32_RD_1_epoch_30iter_223000.pth  \
#              --model_path_p ./ckpts/yuv_input/gan/clip/rt_frame_32_RDL1_repeat_64_epoch_1iter_10000.pth\
#              --test_config ./eval/config_huawei.json \
#              --save_decoded_frame True \
#              --cuda 1 -w 4\
#              --write_stream 1 \
#              --output_path output.json \
#              --cuda_idx 1\
#              --reset_interval 32 \
#              --check_existing 0 \
#              --rate_num 10\
#              --qp_p 4  12 14 16 18 19 20 22 23 30 \
#              --qp_i  4  12 14 16 18 19 20 22 23 30  \
#              --force_intra_period 11111\
#              --calc_ssim True \
#              --run_fast True \
#              --run_fast True \

# CUDA_VISIBLE_DEVICES=7 python -m compress_rtintra \
#     --model_path_i ./ckpts/images/yuv_input/gan/rt_intra_32_RD_1_epoch_50iter_374000.pth\
#     --img_path="/workspace/datasets/Kodak/" \
#     --qp_i 1\


#开始



# python test_video_gan.py --model_path_i /workspace/yx/DCVC/RT/ckpts/images/yuv_input/gan/rt_intra_32_RD_1_epoch_69iter_513000.pth  \
#              --test_config ./config_UVG.json \
#              --save_decoded_frame True \
#              --cuda 1 -w 4\
#              --write_stream 1 \
#              --output_path output.json  \
#              --cuda_idx  2 3 5 7\
#              --reset_interval 32 \
#              --check_existing 0 \
#              --rate_num 10\
#              --qp_i  4  12 14 16 18 19 20 22 23 30 \
#              --force_intra_period 1\
#              --force_frame_num 96 \
#              --force_intra True\
#              --run_fast True \
    
python test_video.py --model_path_i /workspace/yx/DCVC/RT/tempbest_i.pth  \
            --model_path_p /workspace/yx/DCVC/RT/tempbest_p.pth  \
             --test_config ./config_temp.json \
             --save_decoded_frame True \
             --cuda 1 -w 4\
             --write_stream 1 \
             --output_path out_log/output_temp.json \
             --cuda_idx  5\
             --reset_interval 32 \
             --check_existing 0 \
             --rate_num 32\
             --qp_i  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
             --qp_p  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
             --force_intra_period 1000000\