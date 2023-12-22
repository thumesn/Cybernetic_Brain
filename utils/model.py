import torch.nn as nn
import torch
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
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
    
class SNNModel(nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()
        spike_grad = surrogate.fast_sigmoid()
        beta = 0.5
        self.snn = nn.Sequential(nn.Conv2d(3, 12, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(12, 32, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            ) 
        self.fc = nn.Sequential(
                    nn.Linear(512, 100),
                    nn.Linear(100, 2),
                    # 经测试，最后一层的leaky层去掉会使得性能更好；
                    # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    )

    def forward(self, x):
        utils.reset(self.snn)
        utils.reset(self.fc)
        x = self.snn(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y
    
    
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
    

class CorrectNContrast(nn.Module):
    def __init__(self,device='cuda'):
        super(CorrectNContrast, self).__init__()
        self.device = device
        self.net = MyModel().to(device)

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

    def eval(self, x, col, change_col=False):
        return [self.net.pred(x)]

    
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

        self.feature = nn.Sequential(
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

        self.class_classifier = nn.Sequential(
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

        self.domain_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        output_cls = self.class_classifier(feature)
        output_dom = self.domain_classifier(reverse_feature)

        return output_cls, output_dom
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


class MyModel_cnc(nn.Module):
    def __init__(self, device = 'cuda'):
        super(MyModel, self).__init__()
        self.device = device
        self.net = LeNet(3, 2).to(device) 

    def forward(self, x,  target ):
        return self.net(x), target
        
    def eval(self, x ):
        return self.net(x)