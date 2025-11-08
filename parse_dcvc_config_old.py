import json
import os
import pandas as pd  # 导入pandas库
def parse_dcvc_json(path,fps):
    assert path.endswith("json")
    with open(path, "r") as f:
        data = json.load(f)
    bpp = data["ave_p_frame_bpp"]
    psnr_y = data["ave_all_frame_psnr_y"]
    msssim = data["ave_all_frame_msssim"]
    bitrate = bpp * 384 * 256 * fps
    return bitrate, psnr_y, msssim
target_bitrate = 8000  # 改



results = {}
for file in os.listdir("./out_bin/test_seq"): #改
    if file.endswith(".json"):
        basename, q_index, _ = file.split(".")
        bitrate, psnr_y, msssim = parse_dcvc_json(
            f"./out_bin/test_seq/{file}", 8  # 改
        )
        if basename in results.keys():
            if abs(float(results[basename][1]) - target_bitrate) > abs(
                float(bitrate) - target_bitrate
            ):
                results[basename] = [q_index, bitrate, psnr_y, msssim]
        else:
            results[basename] = [q_index, bitrate, psnr_y, msssim]

# 按名称排序
results = sorted(results.items(), key=lambda d: d[0])

print(results)


# results = {}
# for file in os.listdir("./out_bin/8fp"): #改
#     if file.endswith(".json"):
#         basename,q_index,_ = file.split(".")
#         bitrate, psnr_y = parse_dcvc_json(
#             f"./out_bin/8fp/{file}",8  # 改
#         )
#         if basename in results.keys():
#             if abs(float(results[basename][2]) - target_psnr) > abs(
#                 float(psnr_y) - target_psnr
#             ):
#                 results[basename] = [q_index, bitrate, psnr_y]
#         else:
#             results[basename] = [q_index,bitrate, psnr_y]
# results = sorted(results.items(), key=lambda d: d[1][2])
