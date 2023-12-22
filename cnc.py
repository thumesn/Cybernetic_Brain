from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from tqdm import tqdm
import os
import numpy as np
from torchvision import transforms
from torchvision import datasets
from utils.loss import ContrastLoss
from utils.model import CorrectNContrast
from utils.model import MyModel_cnc as MyModel
from utils.dataset import ColoredMNIST_cnc as ColoredMNIST
from utils.dataset import ContrastiveDataset
from utils.utils import set_all_seeds
from utils.eval import eval

device=torch.device("cuda:0")

model_erm = MyModel(device=device)

optimizer_erm = torch.optim.Adam(model_erm.parameters(), lr=1e-4)
criterion=nn.CrossEntropyLoss()

trainDataset = ColoredMNIST()
testDataset = ColoredMNIST( env='test')
train_loader = DataLoader(trainDataset, batch_size=64, shuffle=True)
test_loader = DataLoader(testDataset, batch_size=64, shuffle=False)

for epoch in range(20):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for index, (img, target) in loop:
        
        img, target  = img.to(device), target.to(device) 
        pred, target = model_erm(img, target)
        optimizer_erm.zero_grad()
        acc = (pred.argmax(dim=1) == target).float().mean()
        loss = criterion(pred, target)
        loss.backward()
        optimizer_erm.step()
        loop.set_description(f'In Epoch {epoch}')
        loop.set_postfix(loss=loss.detach().cpu().item(), acc=acc.detach().cpu().item())

    acc = eval(model_erm, device, test_loader)
    print(f"After epoch {epoch}, the accuracy is {acc}")
    save_dir = "./out/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_erm.state_dict(), f"./out/epoch{epoch}.pth")



print('[INFO] Start training CNC ...')

losses_cnc = {
    'contrast': ContrastLoss(0.1),
    'cross_entropy': nn.CrossEntropyLoss(),
    'lambda': 0.5
    } 

model = CorrectNContrast(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
cncDataset = ContrastiveDataset(trainDataset, model_erm, device)
cncLoader = DataLoader(cncDataset, batch_size=16, shuffle=True)



for epoch in range(10):
    loop = tqdm(enumerate(cncLoader), total=len(cncLoader))
    for index, data in loop:
        data = [d.to(device) for d in data]
        pred, target, feat = model(*data)
        acc = (pred.argmax(dim=1) == target).float().mean()
        loss_dict = {
            'contrast': losses_cnc['contrast'](feat),
            'cross_entropy': losses_cnc['cross_entropy'](pred, target),
        }
        loss = loss_dict['contrast'] * losses_cnc['lambda'] + loss_dict['cross_entropy'] * (1 - losses_cnc['lambda'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'In Epoch {epoch}')
        loop.set_postfix(loss=loss.detach().cpu().item(), acc=acc.detach().cpu().item())
    acc = eval(model, device, test_loader)
    print(f"After epoch {epoch}, the accuracy is {acc}")
    torch.save(model.state_dict(), f"./out/cnc_epoch{epoch}.pth")