# RS-semantic-segmentation-pytorch

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]


## Introduction
This repo aims at implementing multiple semantic segmentation models on Pytorch(1.3) for RemoteSensing image datasets.

## Requirements
**Coming Soon.**

## Installation
**Coming Soon.**

## Usage
### Train
```
**Coming Soon.**
```
### Evaluation
```
**Coming Soon.**
```

## Todo
- [ ] Deeplab v3+
- [ ] Unet
- [ ] Linknet
- [ ] Dinknet
- [ ] RGBI images

## Code structure
The code structure is based on [pytorch-template](https://github.com/victoresque/pytorch-template/blob/master/README.md)
  ```
  RS-semantic-segmentation-pytorch/
  │
  ├── train.py - main script to start training
  ├── test.py - test using a trained model
  ├── trainer.py - script for training  
  ├── configs/
  │   ├── xxx.json
  │   ├── xxx.json
  │   └── xxx.json    
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │
  ├── dataloader/ - loading the data for different segmentation datasets
  │
  ├── models/ - contains semantic segmentation models
  │
  ├── saved/
  │   ├── runs/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │  
  └── utils/ - small utility functions
      ├── losses.py - losses used in training the model
      ├── metrics.py - evaluation metrics used
      └── lr_scheduler - learning rate schedulers 
  ```

## Acknowledgement
[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[pytorch-segmentation](https://github.com/yassouali/pytorch_segmentation)

[awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)


[python-image]: https://img.shields.io/badge/Python-3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.3-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/Apache-2.0-blue.svg
[lic-url]: #