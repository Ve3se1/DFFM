import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from numba import njit

# --------- 方法1：OpenCV增强 ----------
def non_dominant(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L, A, B = cv2.split(lab)
    A_eq = cv2.equalizeHist(A)
    B_eq = cv2.equalizeHist(B)
    alpha = 0.1
    A_mix = cv2.addWeighted(A, 1 - alpha, A_eq, alpha, 0)
    B_mix = cv2.addWeighted(B, 1 - alpha, B_eq, alpha, 0)
    lab_mix = cv2.merge([L, A_mix, B_mix])
    img_contrast = cv2.cvtColor(lab_mix, cv2.COLOR_Lab2RGB)
    return img_contrast

# --------- 方法2：Numba加速 dominant ----------
@njit
def retinal_cell(patch):
    center = patch[1, 1]
    gradient_map = np.abs(patch - center)
    g_mean = np.sum(gradient_map) / 8
    theta = g_mean + 3
    ganglion = np.zeros((3, 3), dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            if gradient_map[i, j] < theta:
                ganglion[i, j] = 1
    return ganglion

@njit
def lgn_cell(ganglion):
    score = 0
    if ganglion[1, 1] and ganglion[1, 0] and ganglion[1, 2]:
        score += 1
    if ganglion[1, 1] and ganglion[0, 2] and ganglion[2, 0]:
        score += 1
    if ganglion[1, 1] and ganglion[0, 1] and ganglion[2, 1]:
        score += 1
    if ganglion[1, 1] and ganglion[0, 0] and ganglion[2, 2]:
        score += 1
    return score



@njit
def pad_edge(channel):
    h, w = channel.shape
    padded = np.empty((h + 2, w + 2), dtype=channel.dtype)

    # 中间填充原数组
    for i in range(h):
        for j in range(w):
            padded[i + 1, j + 1] = channel[i, j]

    # 上边缘复制第一行
    for j in range(w):
        padded[0, j + 1] = channel[0, j]

    # 下边缘复制最后一行
    for j in range(w):
        padded[h + 1, j + 1] = channel[h - 1, j]

    # 左边缘复制第一列
    for i in range(h):
        padded[i + 1, 0] = channel[i, 0]

    # 右边缘复制最后一列
    for i in range(h):
        padded[i + 1, w + 1] = channel[i, w - 1]

    # 四个角复制
    padded[0, 0] = channel[0, 0]
    padded[0, w + 1] = channel[0, w - 1]
    padded[h + 1, 0] = channel[h - 1, 0]
    padded[h + 1, w + 1] = channel[h - 1, w - 1]

    return padded


@njit
def process_single_channel(channel):
    h, w = channel.shape
    padded = pad_edge(channel)
    output = channel.copy()
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            patch = padded[i - 1:i + 2, j - 1:j + 2]
            ganglion = retinal_cell(patch)
            score = lgn_cell(ganglion)
            if score == 4:
                output[i - 1, j - 1] = output[i - 1, j - 1] * 0.5
    return output


def dominant_numpy(img_np):
    """ img_np: (H, W, 3) RGB 图像，np.uint8 """
    img_np = img_np.astype(np.float32)
    out = np.zeros_like(img_np)
    for c in range(3):
        out[:, :, c] = process_single_channel(img_np[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)

# --------- 主函数 ----------
def process_images(input_dir, output_cv2_dir, output_torch_dir):
    os.makedirs(output_cv2_dir, exist_ok=True)
    os.makedirs(output_torch_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_dir, filename)

        # 方法1：test
        enhanced_img = non_dominant(input_path)
        save_path_cv2 = os.path.join(output_cv2_dir, filename)
        cv2.imwrite(save_path_cv2, cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))

        # 方法2：dominant_numpy
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img)
        processed = dominant_numpy(img_np)
        save_path_torch = os.path.join(output_torch_dir, filename)
        cv2.imwrite(save_path_torch, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

# --------- 执行入口 ----------
if __name__ == "__main__":
    input_folder = r"D:\pycharm\binoculars_AVS\images"  # ← 你可以改成任意输入文件夹路径
    folder_name = os.path.basename(os.path.normpath(input_folder))
    output_non_dominant = f"non_dominant_{folder_name}"
    output_dominant = f"dominant_{folder_name}"
    process_images(input_folder, output_non_dominant, output_dominant)
