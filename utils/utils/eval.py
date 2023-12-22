import torch as torch
from tqdm import tqdm

def eval(model, device, testLoader):

    loop = tqdm(enumerate(testLoader), total=len(testLoader))
    acc = 0
    num = 0
    for index, (img, target) in loop:
        with torch.no_grad():
            img, target = img.to(device), target.to(device)
            
            pred = model.eval(img )
            pred = torch.argmax(pred, dim = 1)
            acc += torch.sum(pred==target)
            num += len(target)
    return acc/num