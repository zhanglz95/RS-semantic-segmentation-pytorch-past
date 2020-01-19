# RS-semantic-segmentation-pytorch

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]


## Introduction
This repo aims at implementing multiple semantic segmentation models on Pytorch(1.3) for RemoteSensing image datasets.

## Preparation
The code was tested with Anaconda and Python 3.7. And you will do the follow to be ready for running the code:
```
# RS-semantic-segmentation-pytorch dependencies, torch and torchvision are installed by pip.
pip install -r requirements.txt

# or if you are in the mainland of China.
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Then clone the Repo
git clone https://github.com/zhanglz95/RS-semantic-segmentation-pytorch.git
```

## Todo
- [X] Deeplab v3+
- [X] Unet
- [X] Linknet
- [X] Dinknet
- [X] CENet
- [X] R2UNet
- [ ] RGBI images

## Code structure
The code structure is based on [pytorch-template](https://github.com/victoresque/pytorch-template/blob/master/README.md)
  ```
  RS-semantic-segmentation-pytorch/
  │
  ├── train.py - main script to start training
  ├── test.py - test using a trained model
  ├── trainer.py - script for training  
  ├── inference.py - script for inference
  ├── requirements.txt - dependencies for the code  
  ├── data/
  │   └── xxxDatasets/
  │       ├── train/
  │       │   ├── image/ - folder contains image files.
  │       │   └── masks/ - folder contains mask files.
  │       └── valid/
  │           ├── image/ - folder contains image files.
  │           └── masks/ - folder contains mask files.  
  │
  ├── configs/
  │   ├── xxx/ - folder contains multiple json files for batch running
  │   ├── xxx.json - json file for single running.
  │   ├── xxx.json
  │   └── xxx.json    
  │
  ├── base/ - abstract base classes
  │   ├── __init__.py  
  │   ├── base_dataloader.py
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │
  ├── dataloader/ - loading the data for different segmentation datasets
  │   ├── __init__.py 
  │   ├── xxxloader.py
  │   ├── xxxloader.py
  │   └── xxxloader.py
  │
  ├── models/ - contains semantic segmentation models
  │   ├── __init__.py
  │   ├── xxNet.py
  │   ├── xxNet.py
  │   └── xxNet.py
  │
  ├── saved/
  │   └── training_name/ - training name for different training config
  │       └── Time/ - Different time when start training
  │  
  └── utils/ - small utility functions
      ├── __init__.py
      ├── loss.py - losses used in training the model
      ├── metrics.py - evaluation metrics used
      ├── optim.py - load torch.optim
      ├── transfunction.py - utils for inference.py
      └── augmentation.py - augmentation utils
  ```


## Usage
### Train
```
**Coming Soon.**
```
### Evaluation
```
**Coming Soon.**
```

## Acknowledgement
[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[pytorch-segmentation](https://github.com/yassouali/pytorch_segmentation)

[awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)

[DeepGlobe-Road-Extraction-Challenge](https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge)

[python-image]: https://img.shields.io/badge/Python-3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.3-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/Apache-2.0-blue.svg
[lic-url]: #
