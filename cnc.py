from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from tqdm import tqdm
import os
import numpy as np
from torchvision import transforms
from torchvision import datasets
class LeNet(nn.Module):
    def __init__(self, input_channel=1, output_channel=2):
        super(LeNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channel, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, output_channel)
        )

    def forward(self, x):
        return self.net(x)


class MyModel(nn.Module):
    def __init__(self, device = 'cuda'):
        super(MyModel, self).__init__()
        self.device = device
        self.net = LeNet(3, 2).to(device) 

    def forward(self, x,  target ):
        return self.net(x), target
        
    def eval(self, x ):
        return self.net(x)

class ColoredMNIST(datasets.VisionDataset):
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
class LeNet_pred(nn.Module):
    def __init__(self, input_channel=1, output_channel=2):
        super(LeNet_pred, self).__init__()

        self.extract = nn.Sequential(
            nn.Conv2d(input_channel, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
        )
        self.final = nn.Linear(84, output_channel)

    def forward(self, x):
        feat = self.extract(x)
        pred = self.final(feat)
        return pred, feat

    def feat(self, x):
        return self.extract(x)

    def pred(self, x):
        x = self.extract(x)
        return self.final(x)
class CorrectNContrast(nn.Module):
    def __init__(self, input_channel=3, output_channel=2, device='cuda'):
        super(CorrectNContrast, self).__init__()
        self.device = device
        self.net = LeNet_pred(input_channel, output_channel).to(device)

    def forward(self, anchors, positives, negatives, target_anchors, target_positives, target_negatives):
        anchors_flat = anchors.view(-1, *anchors.shape[-3:])
        positives_flat = positives.view(-1, *positives.shape[-3:])
        negatives_flat = negatives.view(-1, *negatives.shape[-3:])
        imgs_all = torch.cat([anchors_flat, positives_flat, negatives_flat], dim=0)
        target_all = torch.cat([target_anchors, target_positives, target_negatives], dim=0).view(-1)

        pred_all = self.net.pred(imgs_all)
        feat_all = self.net.feat(imgs_all)
        feat = {
            'anchor': feat_all[:len(anchors_flat)].reshape(*anchors.shape[:2], -1),
            'positive': feat_all[len(anchors_flat):len(anchors_flat) + len(positives_flat)].reshape(*positives.shape[:2], -1),
            'negative': feat_all[len(anchors_flat) + len(positives_flat):].reshape(*negatives.shape[:2], -1),
        }

        return pred_all, target_all, feat

    def eval(self, x):
        return self.net.pred(x)
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
def eval(model, device, testLoader,  ):

    loop = tqdm(enumerate(testLoader), total=len(testLoader))
    acc = 0
    num = 0
    for index, (img, target, col) in loop:
        with torch.no_grad():
            img, target, col = img.to(device), target.to(device), col.to(device)
            
            pred = model.eval(img )
            pred = torch.argmax(pred, dim = 1)
            acc += torch.sum(pred==target)
            num += len(target)
    return acc/num

if __name__=="__main__":
    losses_cnc = {
    'contrast': ContrastLoss(0.1),
    'cross_entropy': nn.CrossEntropyLoss(),
    'lambda': 0.5
    } 
    device=torch.device("cuda:1")
    model = CorrectNContrast(input_channel=3, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model_erm = MyModel(device=device)
    channel=3
    transform=None
    trainDataset = ColoredMNIST( merge_col= channel == 1, transform=transform)
    testDataset = ColoredMNIST( env='test', merge_col=channel == 1, transform=transform)
    train_loader = DataLoader(trainDataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testDataset, batch_size=64, shuffle=False)
    loss_functions = {}
    loss_functions['default'] = nn.CrossEntropyLoss()
    optimizer_erm = torch.optim.Adam(model_erm.parameters(), lr=1e-4)
    
  
    for epoch in range(20):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for index, (img, target, col) in loop:
            
            img, target  = img.to(device), target.to(device) 
            pred, target = model_erm(img, target)
            optimizer_erm.zero_grad()
            acc = (pred.argmax(dim=1) == target).float().mean()
            loss = loss_functions['default'](pred, target)
            loss.backward()
            optimizer_erm.step()
            loop.set_description(f'In Epoch {epoch}')
            loop.set_postfix(loss=loss.detach().cpu().item(), acc=acc.detach().cpu().item())

        acc = eval(model_erm, device, test_loader)
        print(f"After epoch {epoch}, the accuracy is {acc}")
        save_dir = "./out/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model_erm.state_dict(), f"./out/epoch{epoch}_channel{channel}.pth")
        torch.save(model_erm.state_dict(), f"./out/latest_channel{channel}.pth")
    print('[INFO] Start training CNC ...')
    transform=None
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
        torch.save(model.state_dict(), f"./out/cnc_epoch{epoch}_channel{channel}.pth")
        torch.save(model.state_dict(), f"./out/cnc_latest_channel{channel}.pth")