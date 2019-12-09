import os
import torch
from collections import namedtuple
from pathlib import Path
from PIL import Image
import numpy as np

from base import BaseDataLoader
from base import BaseDataSet
Pair = namedtuple('Pair', ['image', 'mask'])


class Visiontek_rgb_dataset(BaseDataSet):
	def __init__(self, root, augment_config):
		super(Visiontek_rgb_dataset, self).__init__(root, augment_config)
		self.num_classes = 1

	def _correspond(self):
		file_path = namedtuple("file_path", ['original_path', 'labeled_path'])

		images_path = Path("images")
		labeled_path = Path("masks")

		self.file_Paths = file_path(original_path=Path(self.root)/images_path,
									labeled_path=Path(self.root)/labeled_path)

		self.files = [file.name.split('.')[0] for file in self.file_Paths.original_path.glob("*.jpg")]

	def _load_data(self, index):
		imgPath = self.file_Paths.original_path / (self.files[index] + ".jpg")
		maskPath = self.file_Paths.labeled_path / (self.files[index] + ".png")

		image = Image.open(imgPath).convert('RGB')
		mask = Image.open(maskPath).convert('L')
		return Pair(image, mask)
		# return img, mask

class Visiontek_rgb_loader(BaseDataLoader):
	def __init__(self, loader_configs):
		root = loader_configs["data_dir"]
		# fetch configs for datasets
		augment_config = loader_configs["augment"]

		self.dataset = Visiontek_rgb_dataset(root, augment_config)

		super(Visiontek_rgb_loader, self).__init__(self.dataset, **loader_configs["args"])

		