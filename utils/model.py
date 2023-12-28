import torch.nn as nn
import torch
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch.utils import reset
import snntorch as snn
from torch.autograd import Function

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class MySNNModel(nn.Module):
    def __init__(self):
        super(MySNNModel, self).__init__()
        self.beta = 0.5
        self.spike_grad = surrogate.fast_sigmoid()
        self.feature = nn.Sequential(nn.Conv2d(3, 12, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(12, 32, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Flatten(),
            ) 
        self.fc = nn.Sequential(
                nn.Linear(512, 100),
                nn.Linear(100, 2),)

    def forward(self, x):
        reset(self.feature)
        reset(self.fc)
        x = self.feature(x)
        x = x.flatten(start_dim = 1)
        x = self.fc(x)
        return x
    
    
class MyModel_dro(nn.Module):
    def __init__(self):
        super(MyModel_dro, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.flatten()
        return x
    
    
class MyModel_irm(nn.Module):
    def __init__(self):
        super(MyModel_irm, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.flatten()
        return x
    
    
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class MyModel_dann(nn.Module):
    def __init__(self):
        super(MyModel_dann, self).__init__()

        self.feat = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.cls_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

        self.dom_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feat = self.feat(input_data)
        feat = feat.view(-1, 50 * 4 * 4)
        reverse_feat = ReverseLayerF.apply(feat, alpha)
        output_cls = self.cls_classifier(feat)
        output_dom = self.dom_classifier(reverse_feat)

        return output_cls, output_dom


# The CNC model refers to https://github.com/WentDong/AI3610_Project/blob/main/model/CorrectNContrast.py
class CorrectNContrast(nn.Module):
    def __init__(self,device='cuda'):
        super(CorrectNContrast, self).__init__()
        self.device = device
        self.net = MyModel_pred().to(device)

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
    
    
class MyModel_pred(nn.Module):
    def __init__(self):
        super(MyModel_pred, self).__init__()
        
        self.extract = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        )
        self.final = nn.Linear(128, 2)

    def forward(self, x):
        feat = self.extract(x)
        pred = self.final(feat)
        return pred, feat

    def feat(self, x):
        return self.extract(x)

    def pred(self, x):
        
        x = self.extract(x)
        return self.final(x)
    
 
    
class MyModel_cnc(nn.Module):
    def __init__(self, device = 'cuda'):
        super(MyModel_cnc, self).__init__()
        self.device = device
        self.net = MyModel().to(device) 

    def forward(self, x,  target ):
        return self.net(x), target
        
    def eval(self, x ):
        return self.net(x)
