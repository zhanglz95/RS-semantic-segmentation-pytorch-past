from torch.utils.data import Dataset
from pathlib import Path
from collections import namedtuple
from PIL import Image
import torch
from torchvision import transforms as T
from utils import AUG

import numpy as np
Pair = namedtuple('Pair', ['image', 'mask'])

class BaseDataSet(Dataset):
    def __init__(self, root, augment_config):
        """
        Initilize parameters
        root (Path): base root for original data and labeled data.
        augment (Bool): wheter does augmentation or not.
        augment_config (dict): the augmentations configs. 
        """
        self.root = root
        # whether to use augmentation
        self.augment = augment_config["applyAugment"]
        # which augmentation to use
        self.augment_method = augment_config["augment_method"]
        # args for each augmentation
        self.aug_kwargs = augment_config["args"]
        '''
        images root and labels root
        pair.original_path and pair.labeled_path
        '''
        self.root_pair = None
        self.file_names = []
        
    def toTensor(self, pair):

        to_tensor = T.ToTensor()

        return Pair(
            to_tensor(pair.image),
            to_tensor(pair.mask)
            )

    def augmentation(self, pair):
        # TODO add augmentation for data
        if self.augment:
            for aug in self.augment_method.keys():
                if self.augment_method[aug]:
                    method = AUG[aug]
                    pair = method(pair, **self.aug_kwargs[aug])
        return pair

    def _load_data(self, index):
        '''
        Cooperate with __getitem__,This method should be implemented based on the dataset's specilities.

        Aim: Return a pair contains a piece of original data and labeled data which readed from the file path.
        pair = namedtuple('pair', ['original', 'labeled'])
        '''
        raise NotImplementedError

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        '''
        Return a named tuple. pair = (origin = xx, label = xx)
        First: Fetch a pair of original data and labeled data.
        Second: Augment the pair of data in 'train' or 'val' way.
        '''
        if index >= self.__len__() or index < 0:
            raise IndexError
        else:
            pair = self._load_data(index)
            pair = self.augmentation(pair)
            pair = self.toTensor(pair)

        return pair