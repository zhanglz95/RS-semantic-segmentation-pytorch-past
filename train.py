import argparse
import json

import dataloader as D
import models as M
from utils import loss as L
from utils import Logger

# from trainer import Trainer

def main(config, resume):
	logger = Logger(config["name"], "info")

	# make Data Loader
	logger.add_info("Start loading train data.")
	train_loader = getattr(D, config["loader_name"])(config["train_loader"])
	logger.add_info("Start loading valid data.")
	val_loader = getattr(D, config["loader_name"])(config["val_loader"])

	for pair in train_loader:
		print(pair)

	# make model network
	model = getattr(M, config["model"])

	# make loss
	loss = getattr(L, config["loss"])

	# trainer = Trainer(
	# 	model,
	# 	loss,
	# 	config,
	# 	train_loader,
	# 	val_loader,
	# 	logger
	# 	)

	# trainer.train()

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


