import os
import torch
from collections import namedtuple
from pathlib import Path
from PIL import Image
import numpy as np

from base import BaseDataLoader
from base import BaseDataSet
Pair = namedtuple('Pair', ['image', 'mask'])
File_path = namedtuple("File_path", ['original_path', 'labeled_path'])


class Deepglobe_rgb_dataset(BaseDataSet):
	def __init__(self, root, augment_config):
		super(Deepglobe_rgb_dataset, self).__init__(root, augment_config)
		self.num_classes = 1

		self.root_pair = File_path(original_path=Path(self.root)/Path("images"),
									labeled_path=Path(self.root)/Path("masks"))

		self.file_names = [file.name.split('_')[0] for file in self.root_pair.original_path.glob("*.jpg")]

	def _load_data(self, index):
		imgPath = self.root_pair.original_path / (self.file_names[index] + "_sat.jpg")
		maskPath = self.root_pair.labeled_path / (self.file_names[index] + "_mask.png")

		image = Image.open(imgPath).convert('RGB')
		mask = np.array(Image.open(maskPath).convert('L'))
		mask[mask > 128] = 1
		mask = Image.fromarray(mask)
		return Pair(image, mask)

class Deepglobe_rgb_loader(BaseDataLoader):
	def __init__(self, loader_configs):
		root = loader_configs["data_dir"]

		augment_config = loader_configs["augment"]

		self.dataset = Deepglobe_rgb_dataset(root, augment_config)

		super(Deepglobe_rgb_loader, self).__init__(self.dataset, **loader_configs["args"])

		