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

	configs = json.load(open(args.config))

	train_loader = getattr(D, configs["loader_name"])(configs["train_loader"])
	val_loader = getattr(D, configs["loader_name"])(configs["val_loader"])

	model = getattr(M, configs["model"])()

	trainer_config = configs["trainer"]

	trainer = Trainer(trainer_config, model, train_loader, val_loader)
	trainer.train()

if __name__ == '__main__':
	# parser = argparse.ArgumentParser(description='Semantic Segmentation for Training...')

	# parser.add_argument('-c', '--config', default='./configs/visiontek_road_rgb_unet_res101.json', 
	# 	type=str, help='Path to the config file.')
	# parser.add_argument('-r', '--resume', default=None, 
	# 	type=str, help='Path to checkpoint for resume training.')

	
	# if args.resume:
	#  	config = torch.load(args.resume)['config']

	#main(config, args.resume)

	main()
