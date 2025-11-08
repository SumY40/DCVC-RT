import json
import os
config_template = {
    "root_path": "./", # 改
    "test_classes": {
        "test_seq": {
            "test": 1,
            "base_path": "huawei/384x256/8fps", #改
            "src_type": "yuv420",
            "sequences": {
                # "Beauty_1920x1080_120fps_420_8bit_YUV.yuv": {
                #     "width": 512,
                #     "height": 384,
                #     "frames": 600,
                #     "intra_period": 16,
                # },
            },
        }
    },
}
sequences = config_template["test_classes"]["test_seq"]["sequences"]
per_frame = int(384 * 256 * 1.5)
path = "huawei/512x384/8fps" #改
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    file_size = os.path.getsize(file_path)
    frames = file_size // per_frame
    sequences[file] = {
        "width": 384,
        "height": 256,
        "frames": frames,
        "intra_period": 1000,
    }
b = json.dumps(config_template,indent=4)
with open('config_huawei_384x256.json', 'w') as f:
    f.write(b)