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
	def __init__(self, root, augment=True):
		super(Visiontek_rgb_dataset, self).__init__(root, augment)


	def _correspond(self):
		file_path = namedtuple("file_path", ['original_path', 'labeled_path'])

		images_path = Path("images")
		labeled_path = Path("masks")

		self.file_Paths = file_path(original_path=Path(self.root)/images_path,
									labeled_path=Path(self.root)/labeled_path)
		self.images = [file for file in self.file_Paths.original_path.glob("*.jpg")]
		self.masks = [file for file in self.file_Paths.labeled_path.glob("*.png")]

		assert len(self.images) == len(self.masks)

	def _load_data(self, index):
		imgPath = self.images[index]
		maskPath = self.masks[index]

		img = Image.open(imgPath).convert('RGB')
		mask = Image.open(maskPath).convert('L')

		return Pair(np.asarray(img), np.asarray(mask))
		# return img, mask

class Visiontek_rgb_loader(BaseDataLoader):
	def __init__(self, loader_configs):
		root = loader_configs["data_dir"]
		augment = loader_configs["augment"]
		dataset = Visiontek_rgb_dataset(root, augment)
		super(Visiontek_rgb_loader, self).__init__(dataset, **loader_configs["args"])

		