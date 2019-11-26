import argparse
import json

import dataloader as D
import models as M
from utils import loss
from utils import Logger

from trainer import Trainer

def main(config, resume):
	train_logger = Logger(config["name"], "info")

	# make Data Loader
	train_loader = getattr(D, config["loader_name"])(config["train_loader"])
	val_loader = getattr(D, config["loader_name"])(config["valid_loader"])

	# make model network
	model = getattr(M, config["model"])

	# make loss
	loss = None

	trainer = Trainer()

	trainer.train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Semantic Segmentation for Training...')

	parser.add_argument('-c', '--config', default='./configs/visiontek_road_rgb_unet_res101.json', 
		type=str, help='Path to the config file.')
	parser.add_argument('-r', '--resume', default=None, 
		type=str, help='Path to checkpoint for resume training.')

	args = parser.parse_args()

	config = json.load(open(args.config))
	if args.resume:
		config = torch.load(args.resume)['config']

	main(config, args.resume)


