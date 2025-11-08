import torch
state_dict = torch.load("/workspace/yx/DCVC/RT/ckpts/image/yuv_input/rt_intra_64_RD__epoch_4iter_34000.pth", map_location="cpu")
total_params = 0
total_bytes = 0
for name, p in state_dict.items():
    num = p.numel()
    size = p.element_size() * num
    total_params += num
    total_bytes += size
    print(f"{name:<45} shape={str(tuple(p.shape)):<25} dtype={str(p.dtype):<15} size={size/1024:.2f} KB")
print(f"\nTotal parameters: {total_params:,}")
print(f"Total size: {total_bytes/1024/1024:.2f} MB")