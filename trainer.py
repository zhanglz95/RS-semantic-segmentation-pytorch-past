import torch
from torchvision import transforms
from tqdm import tqdm

from base import baseTrainer

class Trainer(baseTrainer):
	def __init__(self, model, loss, config, train_loader, val_loader, train_logger):\
		super(Trainer, self).__init__(model, loss, config, train_loader, val_loader, train_logger)

		
	def _train_epoch(self, epoch):
		pass