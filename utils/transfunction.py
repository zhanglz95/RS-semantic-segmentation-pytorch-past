import torch
import torch.nn.functional as F
import numpy as np
from skimage import color

def tensor2numpy(tensor):
	return tensor.cpu().numpy()

def tensor2mask(tensor):
	tensor = F.softmax(tensor, 0)
	mask = torch.argmax(tensor, 0)
	return mask

def label2rgb(label):
	result = color.label2rgb(label, bg_label=0)
	result = result.transpose(2, 0, 1)
	result = torch.Tensor(result)
	return result
