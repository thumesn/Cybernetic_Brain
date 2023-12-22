import torch
from torch import nn
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