import argparse
import json

import dataloader as D
import models as M
from utils import loss as L
from utils import Logger

from trainer import Trainer

def main():
	parser = argparse.ArgumentParser(description='Semantic Segmentation for Training...')

	parser.add_argument('-c', '--config', default='./configs/visiontek_road_rgb_unet_res101.json', 
		type=str, help='Path to the config file.')

	args = parser.parse_args()
	# load configs from .json
	configs = json.load(open(args.config))
	# initial train and valid loader
	train_loader = getattr(D, configs["loader_name"])(configs["train_loader"])
	val_loader = getattr(D, configs["loader_name"])(configs["val_loader"])
	# initial model
	model = getattr(M, configs["model"])()
	# get trainer configs
	trainer_configs = configs["trainer"]

	trainer = Trainer(
		trainer_configs, 
		model, 
		train_loader, 
		val_loader
		)
	# start training
	trainer.train()

if __name__ == '__main__':
	main()
