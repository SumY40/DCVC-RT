from pathlib import Path

# Vimeo sequences 根目录
sequences_dir = Path("/workspace/datasets/vimeo_LS_New_BT709/sequences")

# 输出 txt 文件路径
output_file = "vimeo_filenames_nosort.txt"

with open(output_file, "w") as f:
    # 遍历所有子目录（视频序列）
    for sequence_dir in sequences_dir.iterdir():
        if sequence_dir.is_dir():
            # 遍历子目录下所有图片文件
            for img_file in sequence_dir.glob("*.png"):
                # 拼接成你要的格式： 00001/0001
                line = f"{sequence_dir.name}/{img_file.stem}\n"
                f.write(line)

print(f"已保存所有文件名到 {output_file}")
