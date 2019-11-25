from torch.utils.data import Dataset
from pathlib import Path
from collections import namedtuple
from PIL import Image

class BaseDataSet(Dataset):
    def __init__(self, root, mode='train', aug_kwargs={}, augment=True, **kwargs):
        """
        Initilize parameters
        root (Path): base root for original data and labeled data.
        augment (Bool): wheter does augmentation or not.
        aug_kwargs (dict): the augmentations which you want to do. 
        example {
            "train":{
                method:parameters
            }
            "val":{
                method:parameters
            }
        }
        """
        self.root = root
        self.train_root = root
        self.label_root = root
        self.mode = mode

        self.augment = augment
        self.aug_kwargs = aug_kwargs

        self.files = []
        self._correspond()

        pass

    def augmentation(self, pair, mode):
        '''
        self.aug_kwargs = {method(function) : {parameters of method}}
        All augmentation methods only accept keyword parameters.
        mode is to just whether the augmentations are done on a dataset for val or train. 
        '''
        if self.aug_kwargs:
            for method in self.aug_kwargs[self.mode].keys():
                pair = method(pair=pair, **(self.aug_kwargs[method]))
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
        '''
        if index >= self.__len__() or index < 0:
            raise IndexError
        else:
            pair = self._load_data(index)
            if self.augment:
                pair = self.augmentation(pair, self.mode)
        return pair

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + '\n'
        fmt_str += f"Length:{self.__len__()}||Root:{self.root}||Mode:{self.mode} \n"
        fmt_str += f"Augmentation method:\n{self.aug_kwargs[self.mode]}"
        #fmt_str += f"Train_root:{self.train_root}, Label_root:{self.label_root} "
