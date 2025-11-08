import json

# 加载 JSON 数据
with open("/workspace/yx/DCVC/RT/out_log/output_myrt_RD_yuv.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 用于统计每个码率点的 bpp 与 psnr 累加和及计数
rate_stats = {}

for ds_name, sequences in data.items():
    for seq_name, rates in sequences.items():
        for rate, metrics in rates.items():
            bpp = metrics.get("ave_all_frame_bpp")
            # psnr = metrics.get("ave_all_frame_dists")
            psnr = metrics.get("ave_all_frame_psnr")
            if bpp is None or psnr is None:
                continue
            if rate not in rate_stats:
                rate_stats[rate] = {"bpp_sum": 0.0, "psnr_sum": 0.0, "count": 0}
            rate_stats[rate]["bpp_sum"] += bpp
            rate_stats[rate]["psnr_sum"] += psnr
            rate_stats[rate]["count"] += 1

# 根据码率点排序后生成平均 bpp 与 psnr 的列表，保留6位小数
avg_bpp_list = [
    round(rate_stats[rate]["bpp_sum"] / rate_stats[rate]["count"], 6)
    for rate in sorted(rate_stats.keys())
]
avg_psnr_list = [
    round(rate_stats[rate]["psnr_sum"] / rate_stats[rate]["count"], 6)
    for rate in sorted(rate_stats.keys())
]

# 输出结果
print("平均 bpp 列表：", avg_bpp_list)
print("平均 PSNR 列表：", avg_psnr_list)