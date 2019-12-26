import argparse
import json
from pathlib import Path
import dataloader as D
import models as M
from utils import loss as L

from trainer import Trainer


def train(config_path):
	print(f"Start training from config file {config_path}.")
	# load configs from .json
	configs = json.load(open(config_path))
	# initial train and valid loader
	train_loader = getattr(D, configs["loader_name"])(configs["train_loader"])
	if configs["trainer"]["val"]:
		val_loader = getattr(D, configs["loader_name"])(configs["val_loader"])
	else:val_loader = None
	# initial model
	num_classes = configs["num_classes"]
	model = getattr(M, configs["model"])(num_classes = num_classes)

	trainer = Trainer(
		configs, 
		model, 
		train_loader, 
		val_loader
		)
	# start training
	trainer.train()

def main():
	parser = argparse.ArgumentParser(description='Semantic Segmentation for Training...')

	parser.add_argument('-c', '--config', default='./configs/visiontek_road_rgb_unet_res101.json', 
		type=str, help='Path to the config file.')
	parser.add_argument('-c_dir', "--config_dir", default=None,
		type=str, help="Dir Path to the config files.")

	args = parser.parse_args()
	if args.config_dir:
		config_dir = Path(args.config_dir)
		config_paths = config_dir.glob("*.json")
		for config in config_paths:
			train(config)
	else:
		train(args.config)

if __name__ == '__main__':
	main()
