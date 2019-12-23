import torch
import numpy as np
from torchvision import transforms, utils
from base import BaseTrainer, DataPrefetcher
from utils.metrics import Metrics
from utils.transfunction import tensor2mask, label2rgb, tensor2numpy
import time
class Trainer(BaseTrainer):
	def __init__(self, config, model, train_loader, val_loader):
		super(Trainer, self).__init__(config, model, train_loader, val_loader)

		if self.device != torch.device('cpu'):
			self.train_loader = DataPrefetcher(train_loader, device=self.device)
			self.val_loader = DataPrefetcher(val_loader, device=self.device)

	def _train_epoch(self, epoch):
		self.model.train()
		total_loss = 0
		cnt = 0

		metrics = Metrics()
		for idx, data in enumerate(self.train_loader, 0):
			images, masks = data
			images = images.to(self.device)
			masks = masks.to(self.device)

			outputs = self.model(images)
			# Metrics
			metrics.update_input(outputs, masks)
			seg_metrics = metrics.metrics_all(self.config["metrics"])

			self.optimizer.zero_grad()
			loss = self.loss(outputs, masks)
			loss.backward()
			self.optimizer.step()

			total_loss += loss
			cnt += 1
			if self.tb_writer:
				self.tb_writer.add_scalars("scalars/Training", {"loss": total_loss / cnt, "lr": self.lr, **seg_metrics}, self.total_iters * epoch + idx)
				start = time.time()
				img_rgb = images[0]
				gt_rgb = label2rgb(tensor2numpy(masks[0])).type_as(img_rgb)
				pre_rgb = label2rgb(tensor2numpy(tensor2mask(outputs[0]))).type_as(img_rgb)

				print(time.time() - start)
				self.tb_writer.add_image("Images/Training", utils.make_grid([img_rgb, gt_rgb, pre_rgb]), self.total_iters * epoch + idx)

				# 不同步！
				# self.tb_writer.add_image("Training/GT", label2rgb(tensor2numpy(masks[0])), self.total_iters * epoch + idx)
				# self.tb_writer.add_image("Training/Pre", label2rgb(tensor2numpy(tensor2mask(outputs[0]))), self.total_iters * epoch + idx)
				# self.tb_writer.add_image("Training/image", images[0], self.total_iters * epoch + idx)
			show_str = f"Training, epoch: {epoch}, Iter: {idx}, lr: {self.lr}, loss: {total_loss / cnt}"
			for key in seg_metrics:
				this_str = f"{key}: {seg_metrics[key]}"
				show_str += (", " + this_str)
			print(show_str)
			
		average_loss = total_loss / cnt
		return average_loss

	def _val_epoch(self, epoch):
		self.model.eval()
		total_loss = 0
		cnt = 0

		metrics = Metrics()

		for idx, data in enumerate(self.val_loader, 0):
			images, masks = data
			images = images.to(self.device)
			masks = masks.to(self.device)

			outputs = self.model(images).detach()

			# Metrics
			metrics.update_input(outputs, masks)
			seg_metrics = metrics.metrics_all(self.config["metrics"])

			if self.tb_writer:
				self.tb_writer.add_scalars("scalars/Validation", {**seg_metrics}, self.total_iters * epoch + idx)
				img_rgb = images[0]
				gt_rgb = label2rgb(tensor2numpy(masks[0])).type_as(img_rgb)
				pre_rgb = label2rgb(tensor2numpy(tensor2mask(outputs[0]))).type_as(img_rgb)

				self.tb_writer.add_image("Images/Validation", utils.make_grid([img_rgb, gt_rgb, pre_rgb]), self.total_iters * epoch + idx)

			show_str = f"Validation, epoch: {epoch}, Iter: {idx}"
			for key in seg_metrics:
				this_str = f"{key}: {seg_metrics[key]}"
				show_str += (", " + this_str)
			print(show_str)

		global_iou = metrics.global_iou()

		return global_iou