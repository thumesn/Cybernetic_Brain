from utils.model import MyModel, MySNNModel
from utils.utils import set_all_seeds, detect_color, reverse_batch
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import ColoredMNIST

# 初始化模型
set_all_seeds(0)
model = MySNNModel()
model.to('cuda')
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 加载数据集
train_dataset = ColoredMNIST(name='train1')
test_dataset = ColoredMNIST(name='test')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

best_acc = 0

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()

    for images, labels in train_loader:
        images = images.to('cuda')
        images_reverse = reverse_batch(images).to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        outputs_reverse = model(images_reverse)
        loss = criterion(outputs, labels)
        loss_reverse = criterion(outputs_reverse, labels)
        loss = (loss + loss_reverse) / 2 # loss加权
        loss.backward()
        optimizer.step()
        
    # 在每个epoch结束后评估模型
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
            acc = torch.sum(outputs.argmax(dim=1)==labels).detach().cpu().numpy()
            correct+=acc
            total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy}')
    
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "./save/snn/mymodel_biloss.pt")

print(f"Best Accuracy: {best_acc}")



