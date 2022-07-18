# import torch.nn.functional as F 
import torch.nn as nn 
import torch 
import numpy  as np
from src.lrp import LRP

def construct_lrp(model, device):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    mean = torch.FloatTensor(mean).reshape(1,-1,1,1).to(device)
    std = torch.FloatTensor(std).reshape(1,-1,1,1).to(device)

    model.to(device)
    layers, rules = construct_lrp_layers_and_rules_for_vgg16(model)
    
    lrp_model = LRP(layers, rules, device=device, mean=mean, std=std)
    return lrp_model
  

def construct_lrp_layers_and_rules_for_vgg16(model):
    layers = [] 
    rules = [] 
    # Rule is z_plus
    for layer in model.features: # Convolution 
        layers.append(layer)
        rules.append({"z_plus":True, "epsilon":1e-6})
    layers.append(model.avgpool)
    rules.append({"z_plus":True, "epsilon":1e-6})
    layers.append(nn.Flatten(start_dim=1))
    rules.append({"z_plus":True, "epsilon":1e-6})

    # Rule is epsilon 
    for layer in model.classifier: # FCL # 3dense
        layers.append(layer)
        rules.append({"z_plus":False, "epsilon":2.5e-1})
    
    return layers, rules