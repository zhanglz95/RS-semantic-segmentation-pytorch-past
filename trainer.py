import torch
from torch.autograd import Variable
import time
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from base import BaseTrainer, DataPrefetcher
from time import time
from utils.metrics import Metrics

class Trainer(BaseTrainer):
	def __init__(self, config, model, train_loader, val_loader):
		super(Trainer, self).__init__(config, model, train_loader, val_loader)
		if self.device == torch.device('cpu'):
			prefetch = False
		else:
			prefetch = True
		if prefetch:
			self.train_loader = DataPrefetcher(train_loader, device=self.device)
			self.val_loader = DataPrefetcher(val_loader, device=self.device)

	def _get_seg_metrics(self, outputs, masks):
		metrics = Metrics(outputs, masks)

		pixAcc = metrics.pixel_accuracy()
		iou = metrics.iou()

		return {
			"pixelAcc": pixAcc.mean(),
			"iou": iou.mean()
		}

	def _train_epoch(self, epoch):
		self.model.train()
		total_loss = 0
		cnt = 0

		tbar = tqdm(self.train_loader, ncols=150)
		for idx, data in enumerate(tbar, 0):
		# for idx, data in enumerate(self.train_loader, 0):
			start_time = time()

			images, masks = data
			images = images.to(self.device)
			masks = masks.to(self.device)

			outputs = self.model(images)

			seg_metrics = self._get_seg_metrics(outputs, masks)
			pixelAcc = seg_metrics.get('pixelAcc')
			iou = seg_metrics.get('iou')

			self.optimizer.zero_grad()
			loss = self.loss(outputs, masks)
			loss.backward()
			self.optimizer.step()

			total_loss += loss
			cnt += 1

			tbar.set_description(f'\nTraining, epoch: {epoch}, Iteration: {idx} || ave_Loss: {total_loss / cnt} || pixelAcc: {pixelAcc:.3f} || iou: {iou:.3f}')

		average_loss = total_loss / cnt
		return average_loss

	def _val_epoch(self, epoch):
		self.model.eval()
		total_loss = 0
		cnt = 0

		tbar = tqdm(self.val_loader, ncols=150)
		for idx, data in enumerate(tbar, 0):
			images, masks = data
			images = images.to(self.device)
			masks = masks.to(self.device)

			outputs = self.model(images).detach()

			loss = self.loss(outputs, masks)
			total_loss += loss
			cnt += 1

			seg_metrics = self._get_seg_metrics(outputs, masks)

			pixelAcc = seg_metrics.get('pixelAcc')

			tbar.set_description(f'\nValdiation, epoch: {epoch+1} Iteration: {1+idx+epoch*len(self.train_loader):8d} || Loss: {total_loss/cnt:.3f} \
								|| pixelAcc: {pixelAcc:.3f}')
		
		return {
			'loss': total_loss / cnt, 
			**seg_metrics
		}