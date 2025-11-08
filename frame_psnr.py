import json
import matplotlib.pyplot as plt

# 假设您的JSON数据存储在名为data.json的文件中
with open('/workspace/lxc/DCVC/RT/output.json', 'r') as file:
    data = json.load(file)
# with open('/workspace/lxc/DCVC/RT/out_log/output_myrt_yuv.json', 'r') as file:
#     data = json.load(file)

if "test_seq"  in data:
    video_data = data["test_seq"]["tonghua3_384x256_8.yuv"]["002"]
elif "YachtRide_1920x1080_120fps_420_8bit_YUV.yuv" in data["UVG"]:
    video_data = data["UVG"]["YachtRide_1920x1080_120fps_420_8bit_YUV.yuv"]["000"]
else:
    video_data = data["UVG"]["ShakeNDry"]["000"]

# 提取帧的 BPP 和 PSNR 值
frame_bpp = video_data["frame_bpp"][1:] # 跳过第一个元素
frame_psnr = video_data["frame_psnr"][1:]

# 生成帧编号列表
frame_numbers = list(range(len(frame_bpp)))

# 创建图形和轴
fig, ax1 = plt.subplots()

# 绘制 BPP 曲线，使用红色
ax1.plot(frame_numbers, frame_bpp, 'r-', label='BPP')
ax1.set_xlabel('Frame Number')
ax1.set_ylabel('BPP', color='r')
ax1.tick_params(axis='y', labelcolor='r')

# 创建第二个 y 轴，共享相同的 x 轴
ax2 = ax1.twinx()
ax2.plot(frame_numbers, frame_psnr, 'b-', label='PSNR')
# ax2.set_ylim(31, 33)  # 设置 PSNR 的 y 轴范围
ax2.set_ylabel('PSNR (dB)', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# 添加标题
plt.title('Frame BPP and PSNR')

# 保存图形到文件
plt.savefig('/workspace/lxc/DCVC/RT/frame.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()