import cv2
import mediapipe as mp
import torch
# 初始化 Mediapipe 的人脸和手部检测模块
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# # 输入和输出文件夹
# input_folder = "/workspace/test/ahuawei/code80M/code80M/secondstage/data/huawei/data/flicker/train"
# output_folder_hand = "/workspace/test/ahuawei/code80M/code80M/secondstage/data/huawei/data/flicker/flicker_train_hand_mask"

# # 创建输出文件夹
# os.makedirs(output_folder_hand, exist_ok=True)

# for filename in tqdm(os.listdir(input_folder), desc='Processing images', unit='file'):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         image_path = os.path.join(input_folder, filename)
#         image = cv2.imread(image_path)

#         # 创建空白 Mask
#         mask_hand = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         mask_hand[:] = 0  # 初始化为全黑


#         # 检测人手
#         with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
#             results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # 获取手部关键点的边界框
#                     x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
#                     y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])
#                     x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1])
#                     y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0])
#                     cv2.rectangle(mask_hand, (x_min, y_min), (x_max, y_max), 255, -1)  # 白色填充手部区域

#         # 保存 Mask 图像
#         mask_hand_output_path = os.path.join(output_folder_hand, filename)
#         cv2.imwrite(mask_hand_output_path, mask_hand)



import mediapipe as mp
import torch
import numpy as np

# 初始化 Mediapipe 的人脸检测模块
mp_face_detection = mp.solutions.face_detection

def get_face_mask(images_tensor):
    """
    从输入图像张量 [B, C, H, W] 中提取人脸区域并生成脸部掩码。

    参数:
    - images_tensor (torch.Tensor): 输入图像张量，尺寸为 [B, C, H, W]，值范围在 [0, 1] 之间

    返回:
    - masks_face (torch.Tensor): 生成的人脸掩码张量，尺寸为 [B, C, H, W]，值为 0 或 1
    """
    # 获取批处理大小
    batch_size = images_tensor.size(0)

    # 创建一个空的掩码张量，尺寸为 [B, C, H, W]，初始为 0
    masks_face = torch.zeros_like(images_tensor, dtype=torch.float32)  # 使用 float32 类型，值为 0 或 1

    # 使用 Mediapipe 处理每一张图片
    with mp_face_detection.FaceDetection(min_detection_confidence=0.9) as face_detection:
        for i in range(batch_size):
            # 获取单张图片的 RGB 格式，并确保它是 uint8 类型
            image_rgb = images_tensor[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C] 格式

            # 将 float32 图像从 [0, 1] 转换为 [0, 255] 的 uint8 图像
            image_rgb = (image_rgb * 255).astype(np.uint8)

            # 使用 Mediapipe 检测人脸
            results = face_detection.process(image_rgb)  # 使用原始的 RGB 图像

            # 如果检测到人脸，更新掩码
            if results.detections:
                for detection in results.detections:
                    # 获取人脸的边界框
                    bboxC = detection.location_data.relative_bounding_box
                    x_min = int(bboxC.xmin * image_rgb.shape[1])
                    y_min = int(bboxC.ymin * image_rgb.shape[0])
                    x_max = int((bboxC.xmin + bboxC.width) * image_rgb.shape[1])
                    y_max = int((bboxC.ymin + bboxC.height) * image_rgb.shape[0])

                    # 用值为 1 填充人脸区域
                    masks_face[i, 0, y_min:y_max, x_min:x_max] = 1  # 假设掩码只使用一个通道
                    masks_face[i, 1, y_min:y_max, x_min:x_max] = 1  # 假设掩码只使用一个通道
                    masks_face[i, 2, y_min:y_max, x_min:x_max] = 1  # 假设掩码只使用一个通道


    return masks_face



# print("所有 Mask 图像已生成并保存到相应的文件夹。")