from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from tqdm import tqdm
import os
import numpy as np
from torchvision import transforms
from torchvision import datasets
from utils.utils import ContrastLoss
from utils.model import CorrectNContrast
from utils.model import MyModel_cnc as MyModel
from utils.dataset import ColoredMNIST_cnc as ColoredMNIST
from utils.dataset import ContrastiveDataset
from utils.utils import set_all_seeds
from utils.utils import eval

set_all_seeds(7)

device=torch.device("cuda")

model_erm = MyModel(device=device)

optimizer_erm = torch.optim.Adam(model_erm.parameters(), lr=1e-4)
criterion=nn.CrossEntropyLoss()

trainDataset = ColoredMNIST()
testDataset = ColoredMNIST( env='test')
train_loader = DataLoader(trainDataset, batch_size=64, shuffle=True)
test_loader = DataLoader(testDataset, batch_size=64, shuffle=False)

num_epochs_erm = 20

for epoch in range(num_epochs_erm):
    for idx, (img, target) in enumerate(train_loader):
        
        img, target  = img.to(device), target.to(device) 
        pred, target = model_erm(img, target)
        optimizer_erm.zero_grad()
        
        loss = criterion(pred, target)
        loss.backward()
        optimizer_erm.step()
    accuracy = eval(model_erm, device, test_loader)
    print(f'Epoch {epoch + 1}/{num_epochs_erm} in erm, Test Accuracy: {accuracy}')

    
print('**********Start training CNC**********')

model = CorrectNContrast(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

criterion_contrast = ContrastLoss(0.1)
criterion_CE = nn.CrossEntropyLoss()
lam = 0.5

cncDataset = ContrastiveDataset(trainDataset, model_erm, device)
cncLoader = DataLoader(cncDataset, batch_size=16, shuffle=True)

best_acc = 0

num_epochs_cnc = 10
for epoch in range(num_epochs_cnc):
    for idx, data in enumerate(cncLoader):
        
        data = [item.to(device) for item in data]
        pred, target, feat = model(*data)
        
        loss_contrast = criterion_contrast(feat)
        loss_CE = criterion_CE(pred, target)
        loss = lam * loss_contrast + (1-lam) * loss_CE
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    accuracy = eval(model, device, test_loader)
    print(f'Epoch {epoch + 1}/{num_epochs_cnc} in cnc, Test Accuracy: {accuracy}')
    
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), f"./save/other_methods/mymodel_cnc.pt")
        
print(f"Best Accuracy: {best_acc}")



