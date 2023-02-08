import torch
import nncf.torch

from nncf.torch import create_compressed_model, load_state
from nncf import NNCFConfig
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision

def load_torchvision_model(model_name, pretrained=False, num_classes=100):
    return torchvision.models.__dict__[model_name](num_classes=num_classes, pretrained=pretrained)

config = NNCFConfig.from_json('conf8_8.json')
model = load_torchvision_model('resnet18', pretrained=False, num_classes=config.get('num_classes', 1000))
model.to('cpu')

resuming_checkpoint = torch.load('resnet18_cifar100_int8_best.pth')
compression_state = resuming_checkpoint['compression_state'] 
compression_ctrl, compressed_model = create_compressed_model(model, config, compression_state=compression_state)
state_dict = resuming_checkpoint['state_dict'] 

load_state(compressed_model, state_dict, is_resume=True)     
compressed_model.load_state_dict(state_dict)

print(compressed_model)
