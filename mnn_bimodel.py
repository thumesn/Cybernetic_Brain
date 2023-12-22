import copy
import memtorch
from memtorch.mn.Module import patch_model
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map
from utils.model import MyModel
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import ColoredMNIST


model_red = MyModel()
model_green = MyModel()
ckpt_red = torch.load(f"./save/cnn/mymodel_red.pt", map_location='cpu')
ckpt_green = torch.load(f"./save/cnn/mymodel_green.pt", map_location='cpu')

model_red.load_state_dict(ckpt_red)
model_green.load_state_dict(ckpt_green)

reference_memristor = memtorch.bh.memristor.VTEAM
reference_memristor_params = {'time_series_resolution': 1e-10}
memristor = reference_memristor(**reference_memristor_params)

patched_model_red = patch_model(copy.deepcopy(model_red),
                          memristor_model=reference_memristor,
                          memristor_model_params=reference_memristor_params,
                          module_parameters_to_patch=[torch.nn.Conv2d],
                          mapping_routine=naive_map,
                          transistor=True,
                          programming_routine=None,
                          tile_shape=(128, 128),
                          max_input_voltage=0.3,
                          scaling_routine=naive_scale,
                          ADC_resolution=8,
                          ADC_overflow_rate=0.,
                          quant_method='linear')

patched_model_green = patch_model(copy.deepcopy(model_green),
                          memristor_model=reference_memristor,
                          memristor_model_params=reference_memristor_params,
                          module_parameters_to_patch=[torch.nn.Conv2d],
                          mapping_routine=naive_map,
                          transistor=True,
                          programming_routine=None,
                          tile_shape=(128, 128),
                          max_input_voltage=0.3,
                          scaling_routine=naive_scale,
                          ADC_resolution=8,
                          ADC_overflow_rate=0.,
                          quant_method='linear')

patched_model_red.tune_()

patched_model_green.tune_()

def test_MyMemResistorModel(model_red, model_green, test_loader):
    model_red.eval()
    model_green.eval()
    correct = 0
    total = len(test_loader.dataset)
    device = 'cpu'
    with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
        for batch_idx, (data, target) in enumerate(test_loader):     
            data, target = data.to(device), target.to(device)  
            output_red = model_red(data)
            output_green = model_green(data)
            outputs = (output_red + output_green) / 2
            pred = outputs.data.max(1)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()
            pbar.update(1)

    accuracy = 100. * float(correct) / float(total)
    return accuracy


test_dataset = ColoredMNIST(name='test')
# To accelerate testing 
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)

print(test_MyMemResistorModel(patched_model_red, patched_model_green, test_loader))

