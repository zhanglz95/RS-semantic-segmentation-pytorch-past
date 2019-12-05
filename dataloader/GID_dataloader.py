import torch
import numpy
from PIL import Image
from pathlib import Path
from collections import namedtuple

from base import BaseDataSet, BaseDataLoader

root = Path('/media/zxpwhu/zxp/datasets/GID/GID/Fine Land-cover Classification_15classes')
Pair = namedtuple('Pair', ['image', 'mask'])

class GID_rgb_dataset(BaseDataSet):
    def __init__(self, root, augment_config):
        super(GID_rgb_dataset, self).__init__(root, augment_config)

    def _correspond(self):
        file_path = namedtuple("file_path", ['original_path', 'labeled_path'])

        images_path = Path('image_RGB')
        labeled_path = Path('label_15classes')

        self.file_Paths = file_path(original_path = Path(self.root)/images_path,
                                    labeled_path = Path(self.root)/labeled_path)



if __name__ == "__main__":

    path = root/"image_RGB"
    for file in path.glob("*.tif"):
        print(file)