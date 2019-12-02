import torch
from torchvision import transforms
from tqdm import tqdm
from base import BaseTrainer, DataPrefetcher
from time import time

class Trainer(BaseTrainer):
	def __init__(self, model, loss, config, train_loader, val_loader, train_logger):
		super(Trainer, self).__init__(model, loss, config, train_loader, val_loader, train_logger)
		
		# I want to log every batch which means self.log_step = 1
		self.log_step = int(self.train_loader.batch_size)
		self.num_classes = self.train_loader.dataset.num_classes
		
		# *************************************************************************
		# # TRANSORMS FOR VISUALIZATION
        # self.restore_transform = transforms.Compose([
        #     local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
        #     transforms.ToPILImage()])
        # self.viz_transform = transforms.Compose([
        #     transforms.Resize((400, 400)),
        #     transforms.ToTensor()])
		# *************************************************************************

		if self.device == torch.device('cpu'):
			prefetch = False
		if prefetch:
			self.train_loader = DataPrefetcher(train_loader, device=self.device)
			self.val_loader = DataPrefetcher(val_loader, device=self.device)

		# https://zhuanlan.zhihu.com/p/73711222
		torch.backends.cudnn.benchmark = True

	def _reset_metrics(self):
		# self.batch_time = AverageMeter()
		# self.data_time = AverageMeter()
		# self.total_loss = AverageMeter()
		self.total_inter, self.total_union = 0, 0
		self.total_correct, self.total_label = 0, 0


	def _update_seg_metrics(self, correct, labeled, inter, union):
		'''
		correct, labeled, intern, union, each one of them is a list which length is num_classes.
		which means that all total_xx are also lists.
		Thus the mean of MIOU indicates calculate "mean" on each class based a batch,
			but not on num_classes based on only one image.
		Summary:
			MIOU for class i = total_inter[i-1] / total_union[i-1]
			union[i] -> the union of (i+1)-th class.
			total_union[i] -> the total union of (i+1)-th class in this batch 
		'''
		
		self.total_correct += correct
		self.total_label += labeled
		self.total_inter += inter
		self.total_union += union 

	def _get_seg_metrics(self):
		pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
		Iou = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
		mIou = Iou.mean()
		# This miou

	def _train_epoch(self, epoch):
		self.logger.info('Training Start!')

		self.model.train()
		mode = 'train'

		# reset all parameters and loss and metrics
		# may be this functino should be implemented in BaseTrainer
		self._reset_metrics()

		tbar = tqdm(self.train_loader, nclos=130)
		for batch_idx, pair in enumerate(tbar):
			# What the difference between data_time and batch_time
			epoch_start = time()
			self.lr_scheduler.step(epoch=epoch-1)

			# loss and optimize
			self.optimizer.zero_grad() # This may should be in reset step
			pre = self.model(pair.image)
			loss = self.loss(pre, pair.mask)
			loss.backward()
			self.optimizer.step()

			epoch_end = time()
			# ******
			# self.total_loss.update(loss.item())
			# ******
			# self.batch_time.update(time() - tic)
			# date_time may used to store all 
			# ******

			# logging and tensorboard
			if batch_idx % self.log_step == 0:
				self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
				self.writer.add_scalar(f'{mode}/loss ', loss.item(), self.wrt_step)

			# evaluation 
			# *********
			seg_metrics = eval_metrics(pre, pair.mask, self.num_classes)
			self._update_seg_metrics(*seg_metrics)
			pixAcc, mIou, _ = self._get_seg_metrics().value()
			
			elapsed_time = epoch_end - epoch_start

			# *********
			# Print Info
			tbar.set_description(f'Train{epoch}||Loss:{loss.item():.2f}||\
					Acc:{pixAcc:.2f} mIou:{mIou:.2f} Time:{elapsed_time:.2f}')

		seg_metrics = self._get_seg_metrics()
		for k, v in list(seg_metrics.items())[:-1]:
			self.writer.add_scalar