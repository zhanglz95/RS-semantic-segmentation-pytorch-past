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
        aug_kwargs (dict): the augmentations which you want to do. 
        example 
        {
            "train":{
                method:parameters
            }
            "val":{
                method:parameters
            }
        }
        """
        self.root = root

        self.augment = augment_config["applyAugment"]
        self.augment_method = augment_config["augment_method"]
        self.aug_kwargs = augment_config["args"]

        self.file_Paths = None
        self.files = []

        self._correspond()
        
    def toTensor(self, pair):
        to_tensor = T.ToTensor()
        return Pair(
            to_tensor(pair.image),
            to_tensor(pair.mask)
            )
    def augmentation(self, pair):
        '''
        self.aug_kwargs = {method(function) : {parameters of method}}
        All augmentation methods only accept keyword parameters.
        mode is to just whether the augmentations are done on a dataset for val or train. 
        '''
        # TODO add augmentation for data


        # if self.aug_kwargs:
        #     for method in self.aug_kwargs[self.mode].keys():
        #         pair = method(pair=pair, **(self.aug_kwargs[self.mode][method]))
        if self.augment:
            for aug in self.augment_method.keys():
                if self.augment_method[aug]:
                    method = AUG[aug]
                    pair = method(pair, **self.aug_kwargs[aug])
        return pair

    def _correspond(self):
        '''
        1.Set original data root and labeled data root. 
        2.Fill self.files with namedtuples each composed by the root of train original data and labeled data.
        
        file_path = namedtuple('file_path', ['original_path', 'labeled_path'])
        
        Tips: Must be called before _load_data
        '''
        raise NotImplementedError

    def _load_data(self, index):
        '''
        Cooperate with __getitem__,This method should be implemented based on the dataset's specilities.

        Aim: Return a pair contains a piece of original data and labeled data which readed from the file path.
        pair = namedtuple('pair', ['original', 'labeled'])
        '''
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

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

    # def __repr__(self):
    #     fmt_str = "Dataset: " + self.__class__.__name__ + '\n'
    #     fmt_str += f"Length:{self.__len__()}||Root:{self.root}||Mode:{self.mode} \n"
    #     fmt_str += f"Augmentation method:\n{self.aug_kwargs[self.mode]}"
    #     #fmt_str += f"Train_root:{self.train_root}, Label_root:{self.label_root} "
