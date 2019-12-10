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
		
		self.num_classes = self.train_loader.dataset.num_classes
		
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



		Iou = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
		mIou = Iou.mean()
		# This miou still calculate mean on classes.
		return {
			"pixelAcc":np.round(pixAcc, 3),
			"mIou":np.round(mIou, 3), 
			"class_Iou":dict(zip(range(self.num_classes), np.round(Iou, 3)))
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

			self.optimizer.zero_grad()
			loss = self.loss(outputs, masks)
			loss.backward()
			self.optimizer.step()

			total_loss += loss
			cnt += 1

			tbar.set_description(f'\nTraining, epoch: {epoch}, Iteration: {idx} || ave_Loss: {total_loss / cnt}')

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
			mIou = seg_metrics.get('mIou')
			pixelAcc = seg_metrics.get('pixelAcc')

			tbar.set_description(f'\nValdiation, epoch: {epoch+1} Iteration: {1+idx+epoch*len(self.train_loader):8d} || Loss: {total_loss/cnt:.3f} \
								|| mIou: {mIou:.3f}  pixelAcc: {pixelAcc:.3f}')

			for k, v in list(seg_metrics.get('class_Iou')):
				self.writer.add_scalar(f'\nValdiation, iou', v) 
		
		return {
			'loss': total_loss / cnt, 
			**seg_metrics
		}