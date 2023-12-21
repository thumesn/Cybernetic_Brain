import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
from utils.build_data import ColoredMNIST
from utils.model import MyModel_dann
from utils.utils import set_all_seeds

@torch.no_grad()
def test(dataloader, dataset, epoch):
    model = torch.load(f"./save/other_methods/mnist_mnistm_model_epoch_{epoch}.pth")
    model = model.eval()
    model = model.to('cuda')

    correct = 0
    total = 0
    
    for idx, (img, target) in enumerate(dataloader):
        img = img.to('cuda')
        target = target.to('cuda')
        
        output_cls, _ = model(input_data=img, alpha = 0)
        pred = output_cls.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total += len(target)

    acc = correct.data.numpy() * 1.0 / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy on {dataset} testset: {acc}')
    return acc

set_all_seeds(0)
model = MyModel_dann()
model.to('cuda')

criterion_cls = nn.CrossEntropyLoss()
criterion_dom = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
    
img_transform_source = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x])),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

target_transform_source = transforms.Compose([
    transforms.Lambda(lambda x: 0 if x < 5 else 1)
])

img_transform_target = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

dataset_source_train = datasets.MNIST(root='../', train=True, transform=img_transform_source, target_transform=target_transform_source, download=False)
dataset_source_test = datasets.MNIST(root='../', train=False, transform=img_transform_source, target_transform=target_transform_source, download=False)
dataset_target_train = ColoredMNIST(transform=img_transform_target, env='train1')
dataset_target_test = ColoredMNIST(transform=img_transform_target, env='test')

dataloader_source_train = DataLoader(dataset_source_train, batch_size=32, shuffle=True)
dataloader_source_test = DataLoader(dataset_source_test, batch_size=32, shuffle=False)
dataloader_target_train = DataLoader(dataset_target_train, batch_size=32, shuffle=True)
dataloader_target_test = DataLoader(dataset_target_test, batch_size=32, shuffle=False)

min_len = min(len(dataloader_source_train), len(dataloader_target_train))

best_acc = 0

num_epochs = 10
for epoch in range(num_epochs):
    for idx, ((img_s, target_s), (img_t, target_t)) in enumerate(zip(dataloader_source_train, dataloader_target_train)):
        if idx >= min_len:
            break
        
        optimizer.zero_grad()
        
        p = float(idx + epoch * min_len) / num_epochs / min_len
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        domain_label_s = torch.zeros(len(target_s)).long()
        domain_label_t = torch.ones(len(target_t)).long()

        img_s = img_s.to('cuda')
        target_s = target_s.to('cuda')
        img_t = img_t.to('cuda')
        domain_label_s = domain_label_s.to('cuda')
        domain_label_t = domain_label_t.to('cuda')

        output_cls, output_dom = model(input_data=img_s, alpha=alpha)
        loss_s_target = criterion_cls(output_cls, target_s)
        loss_s_domain = criterion_dom(output_dom, domain_label_s)

        _, output_dom = model(input_data=img_t, alpha=alpha)
        loss_t_domain = criterion_dom(output_dom, domain_label_t)

        loss = loss_s_domain + loss_s_target + loss_t_domain

        loss.backward()
        optimizer.step()
            
    torch.save(model, f'./save/other_methods/mnist_mnistm_model_epoch_{epoch}.pth')
    
    acc_s = test(dataloader_source_test, 'source', epoch)
    acc_t = test(dataloader_target_test, 'target', epoch)
    
    if acc_t > best_acc:
        best_acc = acc_t
        
print(f"Best Accuracy: {best_acc}")
    
 