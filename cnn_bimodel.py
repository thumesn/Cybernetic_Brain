import os
from utils.model import MyModel
from utils.utils import set_all_seeds
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import ColoredMNIST, RedMNIST, GreenMNIST


model_red = MyModel()
model_green = MyModel()

set_all_seeds(0)

model_red.to('cuda')
model_green.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer_red = optim.Adam(model_red.parameters(), lr=0.001)
optimizer_green = optim.Adam(model_green.parameters(), lr=0.001)

train_dataset_red = RedMNIST(name='train1')
train_dataset_green = GreenMNIST(name='train1')

test_dataset = ColoredMNIST(name='test')
train_loader_red = DataLoader(train_dataset_red, batch_size=64, shuffle=True)
train_loader_green = DataLoader(train_dataset_green, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

ratio_red = len(train_loader_red) / (len(train_loader_red) + len(train_loader_green))
ratio_green = len(train_loader_green) / (len(train_loader_red) + len(train_loader_green))

num_epochs = 50
best_acc = 0

if not os.path.exists('./save/cnn'):
    os.makedirs('./save/cnn')

for epoch in range(num_epochs):
    model_red.train()
    model_green.train()
    for (images_red, targets_red), (images_green, targets_green) in zip(train_loader_red, train_loader_green):

        images_red = images_red.to('cuda')
        images_green = images_green.to('cuda')
        targets_red = targets_red.to('cuda')
        targets_green = targets_green.to('cuda')
        
        optimizer_red.zero_grad()
        optimizer_green.zero_grad()
        outputs_red = model_red(images_red)
        outputs_green = model_green(images_green)

        loss_red = criterion(outputs_red, targets_red)
        loss_green = criterion(outputs_green, targets_green)
        loss_red.backward()
        loss_green.backward()
        optimizer_red.step()
        optimizer_green.step()

    model_red.eval()
    model_green.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to('cuda')
            targets = targets.to('cuda')
        
            outputs_red = model_red(images)
            outputs_green = model_green(images)
            
            outputs = ratio_red * outputs_red + ratio_green * outputs_green
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy}')
    
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model_red.state_dict(), "./save/cnn/mymodel_red.pt")
        torch.save(model_green.state_dict(), "./save/cnn/mymodel_green.pt")
    
print(f"Best Accuracy: {best_acc}")



