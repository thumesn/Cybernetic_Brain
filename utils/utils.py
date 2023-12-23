import torch
from torch import nn
import random
import numpy as np
from tqdm import tqdm

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
    
def reverse_batch(images):
    reverse_images = images.clone()
    reverse_images[:, 0, ...], reverse_images[:, 1, ...] = images[:, 1, ...], images[:, 0, ...]
    return reverse_images
    
@torch.no_grad()
def eval(model, device, testLoader):
    correct = 0
    total = 0
    for idx, (img, target) in enumerate(testLoader):
        img, target = img.to(device), target.to(device)
        pred = model.eval(img )
        pred = torch.argmax(pred, dim = 1)
        correct += torch.sum(pred==target)
        total += len(target)
    accuracy = correct / total
    return accuracy

class ContrastLoss(nn.Module):
    
    def __init__(self, temperature=0.1):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feat):
        anchors, positives, negatives = feat['anchor'], feat['positive'], feat['negative']
        anchor = anchors[:, 0]
        loss_1 = self.loss(anchor, positives, negatives)
        positive = positives[:, 0]
        loss_2 = self.loss(positive, anchors, negatives)
        return loss_1 + loss_2

    def loss(self, anchor, positives, negatives):
        pos = torch.exp(torch.bmm(positives, anchor[..., None]) / self.temperature).squeeze(-1)
        neg = torch.exp(torch.bmm(negatives, anchor[..., None]) / self.temperature).squeeze(-1)
        dominator = pos.sum(dim=1) + neg.sum(dim=1)
        loss = - torch.log(pos).mean(dim=1) + torch.log(dominator)
        return loss.mean()  