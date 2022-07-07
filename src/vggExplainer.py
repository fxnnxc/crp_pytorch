# import torch.nn.functional as F 
import torch.nn as nn 
import torch 
import numpy  as np
from src.lrp import LRP

def construct_vgg16_layers_and_rules(model):
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


def construct_lrp(model, device):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    mean = torch.FloatTensor(mean).reshape(1,-1,1,1).to(device)
    std = torch.FloatTensor(std).reshape(1,-1,1,1).to(device)

    model.to(device)
    layers, rules = construct_vgg16_layers_and_rules(model)
    
    lrp_model = LRP(layers, rules, device=device, mean=mean, std=std)
    return lrp_model
  

class VGGExplaner():
	# Object that captures activation values of CNN while forward pass  
	tensors = [None for i in range(12)]  # activations 
	layer_index = 0


	@staticmethod
	def forward_hook(module, input, output):
		# hook for the CNN layer
		if module.__class__.__name__ == "Conv2d":
			VGGExplaner.tensors.append((input[0], output[0]))
			VGGExplaner.layer_index += 1
		
	def __init__(self, vgg,  preprocess, register_hook=True, device="cuda:0"):
		self.model = vgg
		self.lrp_model = construct_lrp(vgg, device)
		self.preprocess = preprocess
		if register_hook:
			# Register Forwardhook for Convolution Layers
			for m in self.model.modules():
				if m.__class__.__name__ == "Conv2d":
					m.register_forward_hook(VGGExplaner.forward_hook)

	def forward_with_register(self, single_image):
		# forward with activation hooks
		assert single_image.size() == (3, 224,224)
		x = single_image.unsqueeze(0)
		self.reset_static_variables()
		x = self.model(x)		
		return x

	def reset_static_variables(self):
		# clear activation values
		VGGExplaner.tensors = []
		VGGExplaner.layer_index = 0

	def compute_lrp(self, x, y=None, class_specific=False):
		# computation of lrp 
		x = x.unsqueeze(0)
		output = self.lrp_model.forward(x, y=y, class_specific=class_specific)
		all_relevnace = output['all_relevnaces']
		return all_relevnace
