########################################▶︎###############################
###一下是一个简单的读取Colored MNIST例子，请进一步完善。可以进行数据预处理等操作。###
#######################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision import datasets
from torch.utils.data import random_split
from utils.utils import detect_color
import numpy as np
import os 

# 对数据集的分布进行调整
class ColoredMNIST_adjusted(datasets.VisionDataset):    
    def __init__(self, name, balance=True):
        self.data_label = torch.load(f'./ColoredMNIST/{name}.pt')
        self.transform = transforms.ToTensor()
        assert balance in [True, False]
        if balance:
            self.split_data()
            self.prepare_data(self.imgs_red)
            self.prepare_data(self.imgs_green)
            self.data_red = [(self.transform(img) * 255, target) for (img, target) in self.imgs_red]
            self.data_green = [(self.transform(img) * 255, target) for (img, target) in self.imgs_green]
            self.data = self.data_red + self.data_green
            # import pdb; pdb.set_trace()
        else:
            self.data = [(self.transform(img) * 255, target) for (img, target) in self.data_label]
    
    def split_data(self):
        self.imgs_red = [(img, target) for (img, target) in self.data_label if detect_color(img) == 1]
        self.imgs_green = [(img, target) for (img, target) in self.data_label if detect_color(img) == 0]
        
    def prepare_data(self, imgs):
        cnt = np.zeros(2)
        for img, target in imgs:
            cnt[target] += 1
        self.fill(imgs, cnt, 1)
        self.fill(imgs, cnt, 0)
                    
    def fill(self, imgs, cnt, label):
        if (cnt[1 - label] > cnt[label]):
            for img, target in imgs:
                if (target == label):
                    cnt[label] += 1
                    imgs.append((img, target))
                    if (cnt[0] == cnt[1]):
                        return
        return
        
    def __getitem__(self, index):
        img, target = self.data[index]
        return img, target
    
    def __len__(self):
        return len(self.data)  
    

class ColoredMNIST_bjz(datasets.VisionDataset):
    def __init__(self, name):
        self.root = '../'
        self.train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
        self.transform = transforms.ToTensor()
        self.data_label = torch.load(f'./ColoredMNIST/{name}.pt')
        
        assert name in ['train1', 'train2', 'test']
        
        self.cnt = np.zeros((2, 2))
        
        if name == 'train1':
            for idx, (img, target) in enumerate(self.train_mnist):
                if idx >= 20000:
                    break
                binary_label_true = 0 if target < 5 else 1
                colored_img = self.data_label[idx][0]
                color = detect_color(colored_img)
                self.cnt[binary_label_true][color] += 1
        elif name == 'train2':
            for idx, (img, target) in enumerate(self.train_mnist):
                if idx < 20000:
                    continue
                if idx >= 40000:
                    break
                binary_label_true = 0 if target < 5 else 1
                colored_img = self.data_label[idx-20000][0]
                color = detect_color(colored_img)
                self.cnt[binary_label_true][color] += 1
        elif name == 'test':
            for idx, (img, target) in enumerate(self.train_mnist):
                if idx < 40000:
                    continue
                binary_label_true = 0 if target < 5 else 1
                colored_img = self.data_label[idx-40000][0]
                color = detect_color(colored_img)
                self.cnt[binary_label_true][color] += 1
        
        for i in range(2):
            
            if (self.cnt[i][0] > self.cnt[i][1]):
                if name == 'train1':
                    for idx, (img, target) in enumerate(self.train_mnist):
                        if idx >= 20000:
                            break
                        binary_label_true = 0 if target < 5 else 1
                        colored_img = self.data_label[idx][0]
                        label = self.data_label[idx][1]
                        color = detect_color(colored_img)
                        if binary_label_true == i and color == 1:
                            self.data_label.append((colored_img, label))
                            self.cnt[binary_label_true][color] += 1
                            if self.cnt[i][0] == self.cnt[i][1]:
                                break
                elif name == 'train2':
                    for idx, (img, target) in enumerate(self.train_mnist):
                        if idx < 20000:
                            continue
                        if idx >= 40000:
                            break
                        binary_label_true = 0 if target < 5 else 1
                        colored_img = self.data_label[idx-20000][0]
                        label = self.data_label[idx-20000][1]
                        color = detect_color(colored_img)
                        if binary_label_true == i and color == 1:
                            self.data_label.append((colored_img, label))
                            self.cnt[binary_label_true][color] += 1
                            if self.cnt[i][0] == self.cnt[i][1]:
                                break
                elif name == 'test':
                    for idx, (img, target) in enumerate(self.train_mnist):
                        if idx < 40000:
                            continue
                        binary_label_true = 0 if target < 5 else 1
                        colored_img = self.data_label[idx-40000][0]
                        label = self.data_label[idx-40000][1]
                        color = detect_color(colored_img)
                        if binary_label_true == i and color == 1:
                            self.data_label.append((colored_img, label))
                            self.cnt[binary_label_true][color] += 1
                            if self.cnt[i][0] == self.cnt[i][1]:
                                break
                            
            if (self.cnt[i][0] < self.cnt[i][1]):
                if name == 'train1':
                    for idx, (img, target) in enumerate(self.train_mnist):
                        if idx >= 20000:
                            break
                        binary_label_true = 0 if target < 5 else 1
                        colored_img = self.data_label[idx][0]
                        label = self.data_label[idx][1]
                        color = detect_color(colored_img)
                        if binary_label_true == i and color == 0:
                            self.data_label.append((colored_img, label))
                            self.cnt[binary_label_true][color] += 1
                            if self.cnt[i][0] == self.cnt[i][1]:
                                break
                elif name == 'train2':
                    for idx, (img, target) in enumerate(self.train_mnist):
                        if idx < 20000:
                            continue
                        if idx >= 40000:
                            break
                        binary_label_true = 0 if target < 5 else 1
                        colored_img = self.data_label[idx-20000][0]
                        label = self.data_label[idx-20000][1]
                        color = detect_color(colored_img)
                        if binary_label_true == i and color == 0:
                            self.data_label.append((colored_img, label))
                            self.cnt[binary_label_true][color] += 1
                            if self.cnt[i][0] == self.cnt[i][1]:
                                break
                elif name == 'test':
                    for idx, (img, target) in enumerate(self.train_mnist):
                        if idx < 40000:
                            continue
                        binary_label_true = 0 if target < 5 else 1
                        colored_img = self.data_label[idx-40000][0]
                        label = self.data_label[idx-40000][1]
                        color = detect_color(colored_img)
                        if binary_label_true == i and color == 0:
                            self.data_label.append((colored_img, label))
                            self.cnt[binary_label_true][color] += 1
                            if self.cnt[i][0] == self.cnt[i][1]:
                                break
        
    def __getitem__(self, index):
        img, target = self.data_label[index]
        img = self.transform(img) 
        return img, target
    
    def __len__(self):
        return len(self.data_label)

    
class ColoredMNIST(datasets.VisionDataset):
    def __init__(self, name):
        self.data_label = torch.load(f'./ColoredMNIST/{name}.pt')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        img, target = self.data_label[index]
        img = self.transform(img) 
        return img, target
    def __len__(self):
        return len(self.data_label)

# 仅加载红色图片
class RedMNIST(datasets.VisionDataset):
    def __init__(self, name):
        
        self.all_data_label = torch.load(f'./ColoredMNIST/{name}.pt')
        self.data_label = []
        for img, target in self.all_data_label:
            if detect_color(img) == 1:
                self.data_label.append((img, target))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        img, target = self.data_label[index]
        img = self.transform(img) 
        return img, target
    def __len__(self):
        return len(self.data_label)
class ColoredMNIST_cnc(datasets.VisionDataset):
    def __init__(self, root='', env='train1', transform=None, target_transform=None, merge_col = False):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        if env in ['train1', 'train2', 'test']:
            data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        elif env == 'all_train':
            data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                                     torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')
        self.num = [0,0]
        self.col_label = np.zeros((2,2))
        self.data_label_color_tuples = []
        for img, target in data_label_tuples:
            img = transforms.ToTensor()(img)
            col = (torch.sum(img[0])==0)
            self.num[col] += 1
            self.col_label[col][target] += 1
            if merge_col:
                img = img.sum(dim=0).unsqueeze(dim=0)
            self.data_label_color_tuples.append(tuple([img, target, col]))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
        img, target, col = self.data_label_color_tuples[index]
        # img = transforms.ToTensor()(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, col

    def __len__(self):
        return len(self.data_label_color_tuples)   
class ContrastiveDataset(Dataset):
    def __init__(self, data, model, device):
        super(ContrastiveDataset, self).__init__()
        self.data = data
        self.groups = [[], [], [], []]  # 0 for true negative, 1 for false negative, 2 for false positive, 3 for true positive

        bs = 64
        loader = DataLoader(data, batch_size = bs, shuffle = False)
        for i, (img, target, col) in enumerate(loader):
            ids = range(i * bs, min(i * bs + bs, len(data)))
            with torch.no_grad():
                img, target, col = img.to(device), target.to(device), col.to(device)
                pred, _ = model(img, target )
                pred = torch.argmax(pred, dim=1)
                for idx, p, t in zip(ids, pred, target):
                    self.groups[p * 2 + t].append(idx)

        self.group_len = [len(self.groups[i]) for i in range(4)]
        self.output_size = 4

    def __len__(self):
        return self.group_len[0] + self.group_len[3] - 2 * self.output_size + 2

    def __getitem__(self, idx):
        if idx < self.group_len[0] - self.output_size + 1:
            group = 0
            group_positive = 2
            group_negative = 1
            positive = 0
        else:
            idx = idx - self.group_len[0] + self.output_size - 1
            group = 3
            group_positive = 1
            group_negative = 2
            positive = 1

        anchors, positives, negatives = [], [], []
        for i in range(idx, idx + 4):
            # get points just use the image
            anchors.append(self.data[self.groups[group][i % self.group_len[group]]][0])
            positives.append(self.data[self.groups[group_positive][i % self.group_len[group_positive]]][0])
            negatives.append(self.data[self.groups[group_negative][i % self.group_len[group_negative]]][0])

        target_anchors = torch.tensor([positive] * 4)
        target_positives = torch.tensor([positive] * 4)
        target_negatives = torch.tensor([1 - positive] * 4)

        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives), target_anchors, target_positives, target_negatives   
# 仅加载绿色图片    
class GreenMNIST(datasets.VisionDataset):
    def __init__(self, name):
        self.all_data_label = torch.load(f'./ColoredMNIST/{name}.pt')
        self.data_label = []
        for img, target in self.all_data_label:
            if detect_color(img) == 0:
                self.data_label.append((img, target))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        img, target = self.data_label[index]
        img = self.transform(img) 
        return img, target
    def __len__(self):
        return len(self.data_label)