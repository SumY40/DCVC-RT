import torch
from src.models.video_model import DMC

def copy_except_indices(model, source_idx, exclude_indices):
    """
    除了 exclude_indices 之外，把 q_encoder 的所有索引复制成 source_idx 的参数。
    """
    with torch.no_grad():
        for idx in range(model.q_encoder.size(0)):
            if idx not in exclude_indices:
                model.q_encoder[idx] = model.q_encoder[source_idx].clone()
                model.q_decoder[idx] = model.q_decoder[source_idx].clone()
                model.q_feature[idx] = model.q_feature[source_idx].clone()
                model.q_recon[idx] = model.q_recon[source_idx].clone()
def copy_bitestimator_params(bit_estimator, source_idx, exclude_indices):
    with torch.no_grad():
        for submodule_name in ['f1', 'f2', 'f3', 'f4']:
            bitparm = getattr(bit_estimator, submodule_name)

            for param_name in ['h', 'b']:
                param = getattr(bitparm, param_name)
                for idx in range(param.size(0)):
                    if idx not in exclude_indices:
                        param[idx] = param[source_idx].clone()

            # 处理a参数（可能为None）
            if bitparm.a is not None:
                param = bitparm.a
                for idx in range(param.size(0)):
                    if idx not in exclude_indices:
                        param[idx] = param[source_idx].clone()

def main():
    checkpoint_path = "./ckpts/rgb_input/rt_frame_64_RD_fix_qp60_7_lr1e-5.pth"       # 替换成你模型文件路径
    output_path = "./ckpts/rgb_input/rt_frame_64_RD_fix_qp60_7_lr1e-5_1.pth"   # 保存新模型参数路径

    model = DMC()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    source_index = 60
    exclude_indices = [60, 64, 68]

    copy_except_indices(model, source_index, exclude_indices)
  # 复制 bit_estimator_z 内部参数
    copy_bitestimator_params(model.bit_estimator_z, source_index, exclude_indices)
    torch.save(model.state_dict(), output_path)
    print(f"复制完成，保存到 {output_path}")

if __name__ == "__main__":
    main()
