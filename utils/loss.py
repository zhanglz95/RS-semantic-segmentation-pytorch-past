import torch
import torch.nn as nn

import numpy as np
class CrossEntropyLoss(nn.Module):
	def __init__(self):
		super(CrossEntropyLoss, self).__init__()
		self.Loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
	def forward(self, output, target):
		return self.Loss(output, target)	

class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
		self.PREPROCESS = nn.Softmax(dim = 1)

	def forward(self, output, target):
		output = self.PREPROCESS(output)
		
		N = target.size(0)
		smooth = 1e-100

		output_flat = output.view(N, -1)
		target_flat = target.view(N, -1)

		intersection = output_flat * target_flat
		loss = 2 * (intersection.sum(1) + smooth) / (output_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N

		return loss

class MultiClassDiceLoss(nn.Module):
	
	def __init__(self):
		super(MultiClassDiceLoss, self).__init__()
		self.PREPROCESS = nn.Softmax(dim = 1)

	def forward(self, output, target, weights = None):
		output = self.PREPROCESS(output)

		smooth = 1e-10
		target = torch.unsqueeze(target, 1)
		one_hot_target = torch.zeros_like(output).scatter_(1, target, 1)

		N = one_hot_target.shape[0]
		C = one_hot_target.shape[1]

		if weights is None:
			weights = torch.ones(1, C) * (1 / (C - 1))
			weights[0, 0] = 0  # background set to zero

		weights = weights.type_as(output)

		output_flat = output.view(N, C, -1)
		one_hot_target_flat = one_hot_target.view(N, C, -1)

		intersection = output_flat * one_hot_target_flat
		print(intersection)

		loss = 1 - 2 * (intersection.sum(2)) / (output_flat.sum(2) + one_hot_target_flat.sum(2) + smooth)

		loss *= weights

		loss = loss.sum() / (N - 1)

		return loss

class MultiClassBatchDiceLoss(nn.Module):
	'''
	dice loss computed in whole batch and classes, which consider area difference in each image
	'''
	def __init__(self):
		super(MultiClassBatchDiceLoss, self).__init__()
		self.PREPROCESS = nn.Softmax(dim = 1)

	def forward(self, output, target, weights = None):
		output = self.PREPROCESS(output)

		smooth = 1e-10
		target = torch.unsqueeze(target, 1)
		one_hot_target = torch.zeros_like(output).scatter_(1, target, 1)

		N = one_hot_target.shape[0]
		C = one_hot_target.shape[1]

		if weights is None:
			weights = torch.ones(1, C) * (1 / (C - 1))
			weights[0, 0] = 0  # background set to zero

		weights = torch.unsqueeze(weights, 2).type_as(output)

		output_flat = output.view(N, C, -1) * weights
		one_hot_target_flat = one_hot_target.view(N, C, -1) * weights

		intersection = output_flat * one_hot_target_flat

		loss = 1 - 2 * (intersection.sum()) / (output_flat.sum() + one_hot_target_flat.sum() + smooth)
		loss /= (N - 1)

		return loss


class CE_DiceLoss(nn.Module):
	'''
	CrossEntropyLoss + MultiClassDiceLoss
	'''
	def __init__(self):
		super(CE_DiceLoss, self).__init__()
		self.CrossEntropyLoss = CrossEntropyLoss()
		self.DiceLoss = MultiClassDiceLoss()

	def forward(self, output, target):
		return self.CrossEntropyLoss(output, target) + self.DiceLoss(output, target)

class CE_BatchDiceLoss(nn.Module)
	'''
	CrossEntropyLoss + MultiClassBatchDiceLoss
	'''
	def __init__(self):
		super(CE_DiceLoss, self).__init__()
		self.CrossEntropyLoss = CrossEntropyLoss()
		self.DiceLoss = MultiClassBatchDiceLoss()

	def forward(self, output, target):
		return self.CrossEntropyLoss(output, target) + self.DiceLoss(output, target)


if __name__ == "__main__":
	output = torch.FloatTensor(
		[[[[0.5, 0.6, 0.1]],
		  [[0.5, 0.4, 0.9]]],
		 [[[0.4, 0.2, 0.3]],
		  [[0.6, 0.8, 0.7]]]]
		)
	target = torch.LongTensor(
		[[[0, 1, 0]],
		 [[1, 1, 1]]])

	LOSS = MultiClassBatchDiceLoss()
	loss = LOSS(output, target)
	print(loss)