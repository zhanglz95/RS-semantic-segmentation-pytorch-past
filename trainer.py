import torch
from torchvision import transforms
from tqdm import tqdm

class Trainer:
	def __init__(self, model, loss, config, train_loader, valid_loader, logger, tb):
		self.model = model
		self.loss = loss
		self.config = config
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.logger = logger
		self.tb = tb

		