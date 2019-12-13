import torch
import torch.nn as nn

class NLLLoss(nn.Module):
	def __init__(self, weight=None, ignore_index=0, reduction='mean'):
		super(NLLLoss, self).__init__()
		self.loss =  nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
		self.PREPROCESS = nn.LogSoftmax(dim=1)

	def forward(self, output, target):
		output = self.PREPROCESS(output)
		loss = self.loss(output, target)
		return loss

class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
		self.PREPROCESS = nn.Softmax(dim = 1)

	def forward(self, output, target):
		output = self.PREPROCESS(output)
		
		N = target.size(0)
		smooth = 0.

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

		smooth = 0.
		target = torch.unsqueeze(target, 1)
		one_hot_target = torch.zeros_like(output).scatter_(1, target, 1)

		N = one_hot_target.shape[0]
		C = one_hot_target.shape[1]

		if weights is None:
			weights = torch.ones(1, C) * (1 / C)

		weights = weights.t()

		output_flat = output.view(N, C, -1)
		one_hot_target_flat = one_hot_target.view(N, C, -1)

		intersection = output_flat * one_hot_target_flat
		loss = 1 - 2 * (intersection.sum(2) + smooth) / (output_flat.sum(2) + one_hot_target_flat.sum(2) + smooth)

		loss = loss.mm(weights).sum() / N

		return loss

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

	LOSS = MultiClassDiceLoss()
	loss = LOSS(output, target)
	print(loss)

# class dice_bce_loss(nn.Module):
#	 def __init__(self, batch=True):
#		 super(dice_bce_loss, self).__init__()
#		 self.batch = batch
#		 self.bce_loss = nn.BCELoss()
		
#	 def soft_dice_coeff(self, y_true, y_pred):
#		 smooth = 0.0  # may change
#		 if self.batch:
#			 i = torch.sum(y_true)
#			 j = torch.sum(y_pred)
#			 intersection = torch.sum(y_true * y_pred)
#		 else:
#			 i = y_true.sum(1).sum(1).sum(1)
#			 j = y_pred.sum(1).sum(1).sum(1)
#			 intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
#		 score = (2. * intersection + smooth) / (i + j + smooth)
#		 #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
#		 return score.mean()

#	 def soft_dice_loss(self, y_true, y_pred):
#		 loss = 1 - self.soft_dice_coeff(y_true, y_pred)
#		 return loss
		
#	 def __call__(self, y_true, y_pred):
#		 a =  self.bce_loss(y_pred, y_true)
#		 b =  self.soft_dice_loss(y_true, y_pred)
#		 return a + b