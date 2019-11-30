import torch
import time
from torchvision import transforms
from tqdm import tqdm
from base import BaseTrainer
from base import DataPrefetcher
from utils import eval_metrics, AverageMeter

class Trainer(BaseTrainer):
	def __init__(self,model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
		super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader=None, train_logger=None)
		
		self.log_step = self.config['trainer'].get('log_per_iter', self.train_loader.batch_size)
		self.num_classes = self.train_loader.dataset.num_classes

		if self.device == torch.device('cpu'):
			self.prefetch = False
		
		if self.prefetch:
			self.train_loader = DataPrefetcher(self.train_loader, device=self.device)
			self.val_loader = DataPrefetcher(self.val_loader, device=self.device)
	
		# https://zhuanlan.zhihu.com/p/73711222
		torch.backends.cudnn.benchmark = True


		# torch.backends.cudnn.deterministic = True
		# torch.manual_seed() -> set seed for generating random number to a non-deterministic random number in cpu
		# torch.cuda.manual_seed() || torch.cuda.manual_seed_all() -> set seed for GPU
 		# -> torch will use determined algorithm to compute convolution.

	def _train_epoch(self, epoch):
		self.logger.info('\n')

		self.model.train()
		self.wrt_mode = 'train'
		
		tic = time.time()
		self._reset_metric()

		tbar = tqdm(self.train_loader, ncols=130)
		for batch_idx, (data, target) in enumerate(tbar):
			self.data_time.update(time.time() - tic)
			# data, target = data.to(self.device), target.to(device)

			self.lr_scheduler.step(epoch=epoch-1)

			## Loss & Optimize
			self.optimizer.zero_grad()
			output = self.model(data)
			# # ***************************************************
			#  if self.config['arch']['type'][:3] == 'PSP':
            #     assert output[0].size()[2:] == target.size()[1:]
            #     assert output[0].size()[1] == self.num_classes 
            #     loss = self.loss(output[0], target)
            #     loss += self.loss(output[1], target) * 0.4
            #     output = output[0]
            # else:
            #     assert output.size()[2:] == target.size()[1:]
            #     assert output.size()[1] == self.num_classes 
            #     loss = self.loss(output, target)
			# # ***************************************************
			loss = self.loss(output, target)

			# # **********************************************
			# if isinstance(self.loss, torch.nn.DataParallel):
			# 	loss = loss.mean()
			# # **********************************************

			loss.backward()
			self.optimizer.step()
			self.total_loss.update(loss.item())

			# measure elapsed time
			self.batch_time.update(time.time() - tic)
			tic = time.time()

			# logging and tensorboard
			if batch_idx % self.log_step == 0:
				self.wrt_step = (epoch-1) * len(self.train_loader) + batch_idx
				self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

			# for eval
			seg_metrics = eval_metrics(output, target, self.num_classes)
			self._update_seg_metrics(*seg_metrics)
			pixAcc, mIou, _ = self._get_seg_metrics().value()

			# print info
			tabr.set_description(f'TRAIN({epoch}) || Loss: {self.total_loss.average:.3f}\
								 || Acc: {pixAcc:.3f} mIou: {mIou:.3f} || \
									 B: {batch_time.average:.2f} D: {self.data_time.average:.2f}')

		pass