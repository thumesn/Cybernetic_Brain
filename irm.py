from utils.model import MyModel, SNNModel, MyModel_dro, MyModel_irm
from utils.utils import set_all_seeds, detect_color
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import ColoredMNIST
from tqdm import tqdm
import numpy as np
import random
from torch.autograd import grad

def irm_penalty(losses, dummy):
    return (grad(losses[0::2].mean(), dummy, create_graph=True)[0] * grad(losses[1::2].mean(), dummy, create_graph=True)[0]).sum()

def train_test_MyModel_irm():
    set_all_seeds(7)
    model = MyModel_irm()
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataset_1 = ColoredMNIST(name='train1')
    train_dataset_2 = ColoredMNIST(name='train2')
    test_dataset = ColoredMNIST(name='test')
    train_loader_1 = DataLoader(train_dataset_1, batch_size=64, shuffle=True)
    train_loader_2 = DataLoader(train_dataset_2, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    num_epochs = 50
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()  
        train_loaders = [iter(loader) for loader in [train_loader_1, train_loader_2]]  
        dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to('cuda')  
        penalty_multiplier = epoch ** 2
        for loaders_batch in zip(*train_loaders):  
            optimizer.zero_grad()  
            error = 0  
            penalty = 0  

            for loader, (imgs, labels) in zip(train_loaders, loaders_batch):  
                imgs = imgs.to("cuda")  
                labels = labels.to("cuda").float()  
                output = model(imgs)  
                loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, labels, reduction='none') 
                penalty += irm_penalty(loss_erm, dummy_w)  
                error += loss_erm.mean()  

            loss_irm = error + penalty_multiplier * penalty  
            loss_irm.backward()  
            optimizer.step() 
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to('cuda'), target.to('cuda').float()
                output = model(data)
                # import pdb; pdb.set_trace()
                pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to('cuda')),
                                    torch.Tensor([1.0]).to('cuda'),
                                    torch.Tensor([0.0]).to('cuda'))  
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc = 100. * correct / len(test_loader.dataset)
        if(test_acc > best_acc):
            best_acc = test_acc
            torch.save(model.state_dict(), "./save/other_methods/mymodel_irm.pt")
        print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {test_acc}')
    print(f"Best Test Accuracy: {best_acc}")
    
train_test_MyModel_irm()