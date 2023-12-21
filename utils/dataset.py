########################################▶︎###############################
###一下是一个简单的读取Colored MNIST例子，请进一步完善。可以进行数据预处理等操作。###
#######################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision import datasets
from torch.utils.data import random_split
from utils.utils import detect_color
import numpy as np

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