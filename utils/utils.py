import torch
import random
import numpy as np

# 固定训练的随机种子
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
# 检测图像的颜色
def detect_color(image):
    img_array = np.array(image)
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    red_ratio = np.count_nonzero(red_channel) / red_channel.size
    green_ratio = np.count_nonzero(green_channel) / green_channel.size
    if green_ratio > 0:
        return 0  # 含有绿色像素，返回0
    elif red_ratio > 0:
        return 1  # 含有红色像素，返回1
    else:
        raise AssertionError("图像中没有检测到绿色或红色像素")