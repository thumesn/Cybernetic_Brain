from utils.model import MyModel, SNNModel, MyModel_dro
from utils.utils import set_all_seeds, detect_color
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import ColoredMNIST_adjusted as ColoredMNIST
from tqdm import tqdm
import numpy as np
import random

def divide_dataset(dataset):
    group_names = ['red0', 'red1', 'green0', 'green1']
    grouped_dataset = {}
    for group_name in group_names:
        grouped_dataset[group_name] = []
    with tqdm(total=len(dataset), desc='Grouping Dataset', unit='item') as pbar:
        for image, target in dataset:
            color = detect_color(image.permute(1,2,0))
            if color == 1 and target == 0:
                grouped_dataset['red0'].append((image, target))
            elif color == 1 and target == 1:
                grouped_dataset['red1'].append((image, target))
            elif color == 0 and target == 0:
                grouped_dataset['green0'].append((image, target))
            elif color == 0 and target == 1:
                grouped_dataset['green1'].append((image, target))
            pbar.update(1)
    return grouped_dataset
    
train_dataset = ColoredMNIST(name='train1')
test_dataset = ColoredMNIST(name='test')
train_dataset_grouped = divide_dataset(train_dataset)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train_dro(model, train_dataset_grouped,l2_lambd=.001, etaq=.001):
    set_all_seeds(0)
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    group_names = ['red0', 'red1', 'green0', 'green1']
    q = np.repeat(1/4,4)
    for epoch in tqdm(range(10000)):
        optimizer.zero_grad()
        idx = random.sample(range(4), 1)[0]
        group = group_names[idx]
        
        image, target = random.sample(train_dataset_grouped[group],1)[0]
        
        image = image.to('cuda')
        target = torch.tensor(target).unsqueeze(0).to('cuda')
        target = target.to(torch.float32)
        
        output = model(image)
        
        #L2 regularization loss
        loss = F.binary_cross_entropy_with_logits(output, target)
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss += l2_lambd * l2_norm
        
        #Update q
        lossNP = loss.detach().cpu().numpy()
        q[idx] = q[idx] * np.exp(etaq*lossNP)
        q /= q.sum()
        
        #Update theta
        for g in optimizer.param_groups:
            g['lr'] = 0.001 * q[idx] #Change learning rate
        loss.backward()
        optimizer.step()
        
    return model

@torch.no_grad()
def test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda').float()
            output = model(data)
            pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to('cuda')),
                                torch.Tensor([1.0]).to('cuda'),
                                torch.Tensor([0.0]).to('cuda'))  
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc

model = MyModel_dro()

model = train_dro(model, train_dataset_grouped)

print(test(model))

torch.save(model.state_dict(), './save/other_methods/mymodel_dro.pt')
