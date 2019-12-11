import torch
import torch.nn as nn

class NLLLoss(nn.Module):
	def __init__(self, weight=None, ignore_index=0, reduction='mean'):
		super(NLLLoss, self).__init__()
		self.loss =  nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

	def forward(self, output, target):
		loss = self.loss(output, target)
		return loss

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