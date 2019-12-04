import torch
import time
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from base import BaseTrainer, DataPrefetcher
from time import time
from utils import eval_metrics
class Trainer(BaseTrainer):
	def __init__(self, model, loss, config, train_loader, val_loader, train_logger):
		super(Trainer, self).__init__(model, loss, config, train_loader, val_loader, train_logger)
		
		# I want to log every batch which means self.log_step = 1
		self.log_step = int(self.train_loader.batch_size)
		self.num_classes = self.train_loader.dataset.num_classes
		
		if self.device == torch.device('cpu'):
			prefetch = False
		if prefetch:
			self.train_loader = DataPrefetcher(train_loader, device=self.device)
			self.val_loader = DataPrefetcher(val_loader, device=self.device)

		# https://zhuanlan.zhihu.com/p/73711222
		torch.backends.cudnn.benchmark = True

	def _reset_metrics(self):
		self.clock = time()
		self.total_inter, self.total_union = 0, 0
		self.total_correct, self.total_label = 0, 0


	# def _update_seg_metrics(self, correct, labeled, inter, union):
	# 	'''
	# 	correct, labeled, intern, union, each one of them is a list which length is num_classes.
	# 	Summary:
	# 		1.IOU for class i = total_inter[i-1] / total_union[i-1]
	# 		2.union[i] -> the union of (i+1)-th class.
	# 		3.total_union[i] -> the total union of (i+1)-th class in this batch 
	# 	'''
		
	# 	self.total_correct += correct
	# 	self.total_label += labeled
	# 	self.total_inter += inter
	# 	self.total_union += union
 
	def _update_seg_metrics(self, **seg_metrics):
		pass


# !!!!!This fucntion is unused
	def _get_seg_metrics(self, outputs, masks):
		pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
		Iou = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
		mIou = Iou.mean()
		# This miou still calculate mean on classes.
		return {
			"pixelAcc":np.round(pixAcc, 3),
			"mIou":np.round(mIou, 3), 
			"class_Iou":dict(zip(range(self.num_classes), np.round(Iou, 3))),
			
		}

	def _train_epoch(self, epoch):
		self.logger.info('Training Start!')
		self.model.train()
		total_loss = 0
		cnt = 0

		tbar = tqdm(self.train_loader, ncols=150)
		for idx, data in enumerate(tbar, 0):
			start_time = time()

			images, masks = data[0].to(self.device), data[1].to(self.device)
			outputs = self.model(images)

			self.optimizer.lr = self.lr_scheduler(epoch)
			self.optimizer.zero_grad()

			loss = self.loss(outputs, masks)
			loss.backward()
			self.optimizer.step()
			total_loss += loss
			cnt += 1
			seg_metrics = self._get_seg_metrics(outputs, masks, self.num_classes)
			mIou = seg_metrics.get('mIou')
			pixelAcc = seg_metrics.get('pixelAcc')
			clock = time()
			if idx % self.log_per_iter == 0:
				self.logger.info(f'\nTraining,Cost Time:{clock-start_time:.2f}, epoch: {epoch+1} Iteration: {1+idx+epoch*len(self.train_loader):8d} || Loss: {total_loss/cnt:.3f} \
								|| mIou: {mIou:.3f}  pixelAcc: {pixelAcc:.3f} ')
			tbar.set_description(f'\nTraining, epoch: {epoch+1} Iteration: {1+idx+epoch*len(self.train_loader):8d} || Loss: {total_loss/cnt:.3f} \
								|| mIou: {mIou:.3f}  pixelAcc: {pixelAcc:.3f}')

			for k, v in list(seg_metrics.get('class_Iou')):
				self.writer.add_scalar(f'Training, iou of {self.classes[k]}', v) 
		
		return {
			'loss': total_loss / cnt, 
			**seg_metrics
		}

	def _val_epoch(self, epoch):
		self.logger.info('Training Start!')
		self.model.eval()
		total_loss = 0
		cnt = 0

		tbar = tqdm(self.val_loader, ncols=150)
		for idx, data in enumerate(tbar, 0):
			images, masks = data
			outputs = self.model(images)

			loss = self.loss(outputs, masks)
			total_loss += loss
			cnt += 1

			seg_metrics = self._get_seg_metrics(outputs, masks, self.num_classes)
			mIou = seg_metrics.get('mIou')
			pixelAcc = seg_metrics.get('pixelAcc')

			if idx % self.log_per_iter == 0:
				self.logger.info(f'\nValdiation, epoch: {epoch+1} Iteration: {1+idx+epoch*len(self.train_loader):8d} || Loss: {total_loss/cnt:.3f} \
								|| mIou: {mIou:.3f}  pixelAcc: {pixelAcc:.3f}')

			tbar.set_description(f'\nValdiation, epoch: {epoch+1} Iteration: {1+idx+epoch*len(self.train_loader):8d} || Loss: {total_loss/cnt:.3f} \
								|| mIou: {mIou:.3f}  pixelAcc: {pixelAcc:.3f}')

			for k, v in list(seg_metrics.get('class_Iou')):
				self.writer.add_scalar(f'\nValdiation, iou of {self.classes[k]}', v) 
		
		return {
			'loss': total_loss / cnt, 
			**seg_metrics
		}