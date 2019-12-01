import logging
import json
import math
import torch
import datetime
import torch.nn as nn
from utils import lr_scheduler
from utils import logger
from pathlib import Path
from torch. utils import tensorboard

def get_instance(module, config, name, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class BaseTrainer:

    def __init__(self, model, loss, config, train_loader, val_loader=None, train_logger=None):
        self.model = model
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        self.imporved = False
        self.not_improved_count = 0


        # SETTING DEVICES
        self.device, available_gpus = self._get_available_devices()
        # **************************************************
        # multi gpus parallel computing 
        # if config["use_synch_bn"]:
        #     self.model = convert_mode(self.mode)
        #     self.model = DaraParallelWithCallback(self.model, device_ids=available_gpus)
        # else:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
        # *************************************************
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)  
        self.model.to(self.device)


        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']


        # OPTIMIZER
        # **************************************************
        # if self.config['optimizer']['differential_lr']:
        #     if isinstance(self.model, torch.nn.DataParallel):
        #         trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.module.get_decoder_params())},
        #                             {'params': filter(lambda p:p.requires_grad, self.model.module.get_backbone_params()), 
        #                             'lr': config['optimizer']['args']['lr'] / 10}]
        #     else:
        #         trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.get_decoder_params())},
        #                             {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()), 
        #                             'lr': config['optimizer']['args']['lr'] / 10}]
        # **************************************************
        trainable_params = filter(lambda p:p.requires_grad, self.model.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        # different lr_scheduler class in torch.optim.lr_scheduler ask for various parameters for initializing
        self.lr_scheduler = get_instance(torch.optim.lr_scheduler,'lr_scheduler', config)
        # Deep lab original code as below.
        # self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler']['type'])(self.optimizer, self.epochs, len(train_loader))


        # MONITORING
        if cfg_trainer['monitor']:
            self.mnt_mode = cfg_trainer['monitor']['mode']
            self.mnt_metric = cfg_trainer['monitor']['metric']
            self.mnt_best = -math.inf if self.mnt_mode=='max' else math.inf
            self.early_stopping = cfg_trainer.get('early_stop', math.inf)
        else:
            self.mnt_mode = 'off'
            self.mnt_best = 0
        

        # CHECKPOINTS & TENSORBOARD
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        ## The name for cehckpoint should just for ourselves.
        self.checkpoint_dir = Path(cfg_trainer['save_dir']).joinpath(self.config['name'], start_time)

        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir()

        config_save_path = self.checkpoint_dir.joinpath('config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = Path(cfg_trainer['log_dir']).joinpath(self.config['name'], start_time)
        self.writer = tensorboard.SummaryWriter(writer_dir)

        if resume : # resume defined in configs
            self._resume_checkpoint(resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            results = self._train_epoch(epoch)
            if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)

            #   logging info
            self.logger.info(f'*** Info for epoch {epoch} ***')
            for k, v in results.item():
                self.logger.info(f'*** {str(k):15s} {v}')

            if self.train_logger is not None:
                log = {'epoch' : epoch, **results}
                self.train_logger.add_info(log)

            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
            if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                try:
                    if self.mnt_mode == 'min':
                        # The code below should just base on the way how computing mnt_metric. 
                        self.imporved = (log[self.mnt_metric] < self.mnt_best)
                    else:
                        self.imporved = (log[self.mnt_best] > self.mnt_best)
                except KeyError:
                        self.logger.warning(f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                        break
            
            if self.imporved:
                self.mnt_best = log[self.mnt_metric]
                self.not_improved_count = 0
            else:
                self.not_improved_count += 1

            if self.not_improved_count > self.early_stopping:
                self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                self.logger.warning('Training Stoped')
                break

            # Save Checkpoint
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.imporved)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError

    def _get_available_devices(self):
        sys_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if sys_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu}')
        available_gpus = list(range(sys_gpu))
        return device, available_gpus 

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.__name__,
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = Path(self.checkpoint_dir).joinpath(f'checkpoint-epoch{epoch}.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...') 
        torch.save(state, filename)

        if save_best:
            filename = Path(self.checkpoint_dir).joinpath(f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")
   
   
    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        # CHECK ARGS
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # if self.lr_scheduler:
        #     self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.train_logger = checkpoint['logger']
        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')