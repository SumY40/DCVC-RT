#!/bin/bash

# --- 脚本配置 ---
USERNAME="221180025@nju.edu.cn"
BASE_URL="https://box.nju.edu.cn/seafdav/Generative%20Coding%20Multiple%20Bitrate%20Test%20Results"
# 本地文件基础目录 (所有要上传的文件直接位于此目录下)
LOCAL_BASE_DIR="/workspace/yx/DCVC/RT/out_bin/UVG"

# 提示用户输入密码
echo "请输入 WebDAV 密码，密码输入后将自动开始所有文件的上传："
read -s PASSWORD

# 检查密码是否为空
if [ -z "$PASSWORD" ]; then
    echo "密码不能为空，脚本终止。"
    exit 1
fi

# 完整的用户认证字符串 (包括密码)
AUTH_STRING="${USERNAME}:${PASSWORD}"

echo "--- 开始批量上传文件 ($LOCAL_BASE_DIR) ---"

# --- 要上传的文件列表 (每行包含 远程目标路径/文件名) ---
# 格式：[远程目标目录]/[本地文件名]
UPLOAD_LIST=(
    # Astro Files
    "astro/astro_1920x1080_yuv420p_63kbps.yuv_q0_63kbps.yuv"
    "astro/astro_1920x1080_yuv420p_191kbps.yuv_q12_191kbps.yuv"
    "astro/astro_1920x1080_yuv420p_379kbps.yuv_q20_379kbps.yuv"
    "astro/astro_1920x1080_yuv420p_778kbps.yuv_q31_778kbps.yuv"
    "astro/astro_1920x1080_yuv420p.yuv_q0.bin"
    "astro/astro_1920x1080_yuv420p.yuv_q12.bin"
    "astro/astro_1920x1080_yuv420p.yuv_q20.bin"
    "astro/astro_1920x1080_yuv420p.yuv_q31.bin"

    # Car Files
    "car/car_1920x1080_yuv420p_102kbps.yuv_q0_102kbps.yuv"
    "car/car_1920x1080_yuv420p_197kbps.yuv_q9_197kbps.yuv"
    "car/car_1920x1080_yuv420p_391kbps.yuv_q18_391kbps.yuv"
    "car/car_1920x1080_yuv420p_754kbps.yuv_q27_754kbps.yuv"
    "car/car_1920x1080_yuv420p_950kbps.yuv_q31_950kbps.yuv"
    "car/car_1920x1080_yuv420p.yuv_q0.bin"
    "car/car_1920x1080_yuv420p.yuv_q9.bin"
    "car/car_1920x1080_yuv420p.yuv_q18.bin"
    "car/car_1920x1080_yuv420p.yuv_q27.bin"
    "car/car_1920x1080_yuv420p.yuv_q31.bin"

    # Factory Files
    "factory/factory_1920x1080_yuv420p_52kbps.yuv_q0_52kbps.yuv"
    "factory/factory_1920x1080_yuv420p_205kbps.yuv_q16_205kbps.yuv"
    "factory/factory_1920x1080_yuv420p_368kbps.yuv_q21_368kbps.yuv"
    "factory/factory_1920x1080_yuv420p_746kbps.yuv_q27_746kbps.yuv"
    "factory/factory_1920x1080_yuv420p_1024kbps.yuv_q30_1024kbps.yuv"
    "factory/factory_1920x1080_yuv420p.yuv_q0.bin"
    "factory/factory_1920x1080_yuv420p.yuv_q16.bin"
    "factory/factory_1920x1080_yuv420p.yuv_q21.bin"
    "factory/factory_1920x1080_yuv420p.yuv_q27.bin"
    "factory/factory_1920x1080_yuv420p.yuv_q30.bin"

    # Four Files
    "Four/Four_1920x1080_yuv420p_66kbps.yuv_q0_66kbps.yuv"
    "Four/Four_1920x1080_yuv420p_195kbps.yuv_q12_195kbps.yuv"
    "Four/Four_1920x1080_yuv420p_380kbps.yuv_q20_380kbps.yuv"
    "Four/Four_1920x1080_yuv420p_776kbps.yuv_q30_776kbps.yuv"
    "Four/Four_1920x1080_yuv420p.yuv_q0.bin"
    "Four/Four_1920x1080_yuv420p.yuv_q12.bin"
    "Four/Four_1920x1080_yuv420p.yuv_q20.bin"
    "Four/Four_1920x1080_yuv420p.yuv_q30.bin"
)

# 循环遍历列表并执行上传
for REMOTE_FILE_PATH in "${UPLOAD_LIST[@]}"; do
    # 提取文件名（不含远程目录）
    FILENAME=$(basename "${REMOTE_FILE_PATH}")
    
    # 构造本地绝对路径
    LOCAL_FILE="${LOCAL_BASE_DIR}/${FILENAME}"
    
    # 构造远程上传 URL
    REMOTE_URL="${BASE_URL}/${REMOTE_FILE_PATH}"

    echo ""
    echo "=================================================="
    echo "正在上传: ${FILENAME} -> ${REMOTE_FILE_PATH}"
    echo "=================================================="

    # 检查本地文件是否存在
    if [ ! -f "${LOCAL_FILE}" ]; then
        echo "⚠️ 错误：本地文件未找到：${LOCAL_FILE}"
        echo "---"
        continue
    fi
    
    # 执行 curl 命令，不使用 -s (静默模式)，保留进度条
    curl -T "${LOCAL_FILE}" -u "${AUTH_STRING}" "${REMOTE_URL}" -S
    
    # 检查 curl 命令的退出状态
    if [ $? -eq 0 ]; then
        echo "" # 确保进度条后换行
        echo "✅ 上传成功：${FILENAME}"
    else
        echo "" # 确保进度条后换行
        echo "❌ 上传失败：${FILENAME}"
    fi
    echo "---"
done

echo "🎉 所有文件上传任务完成。"