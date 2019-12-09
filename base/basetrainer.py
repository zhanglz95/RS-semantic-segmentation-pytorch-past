import logging
import torch
import utils
import json
import tensorboardX as tb
from pathlib import Path
from datetime import datetime

def create_object(module, name, **args):
    return getattr(module, name)(**args)

class BaseTrainer:
    def __init__(self, config, model, train_loader, val_loader=None):
        self.model = model
        self.config = config
        # init iter args
        self.start_epoch = 1
        self.epochs = config['epochs']
        self.train_loader = train_loader
        self.log_per_iter = self.config['log_per_iter']
        self.val = self.config['val']
        self.val_loader = val_loader
        self.val_per_epochs = self.config['val_per_epochs']
        
        self.logger = logging.getLogger(self.__class__.__name__)

        self.loss = create_object(utils.loss, self.config['loss'])
        # init optim
        params = filter(lambda p:p.requires_grad, self.model.parameters())
        optim_config = self.config["optimizer"]
        self.optimizer = create_object(utils.optim, optim_config["type"], 
                                        **{'params': params,
                                            'lr':optim_config["lr"]})
       # init metrics
        self.improved = False
        self.best_loss = float("inf")
        self.break_for_grad_vanish = self.config['break_for_grad_vanish']
        # self.moniter = config['moniter']

        # Set Device
        if self.config["use_gpu"]:
            self.available_gpus = list(range(torch.cuda.device_count()))
            self.device = torch.device('cuda:0' if self.available_gpus else 'cpu')
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # https://zhuanlan.zhihu.com/p/73711222
        '''
        improve speed of GPU when network and datasize fixed 
        '''
        torch.backends.cudnn.benchmark = self.config["cudnnBenchmarkFlag"]

        # CheckPoint and Tensorboard
        self.save_per_epochs = self.config['save_per_epochs']
        start_time = datetime.now().strftime("%m-%d-%H:%M")
        self.checkpoints_path = Path(self.config['save_dir']) / (self.config['name']) / (str(start_time))
        if not self.checkpoints_path.exists():
            self.checkpoints_path.mkdir(parents=True)
        config_save_path = self.checkpoints_path/"config.json"
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = Path(self.config['save_dir']) / self.config['name'] / str(start_time)
        self.writer = tb.SummaryWriter(writer_dir)

        if self.config['resume_path']:
            self._resume_checkpoint(self.config['resume_path'])

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            results = self._train_epoch(epoch)
            if self.val and epoch % self.val_per_epochs == 0:
                results = self._val_epoch(epoch)

            # logging info
            self.logger.info(f'\n Info for epoch: {epoch}')
            for key, value in results.items():
                self.logger.info(f'\n{str(key):15s}: {value}')

            # Check if this is the best model
            # self.improved = results[self.moniter] > self.best
            self.improved = results["loss"] < self.best
            if self.improved:
                # self.best = results[self.moniter]
                self.improved = results["loss"]
                self.not_imporved_count = 0
            else:
                self.not_imporved_count += 1

            if self.not_imporved_count > self.brake_for_grad_vanish:
                self.logger.info(f'\nThis Model has not improved for {self.not_imporved_count} epochs.\n \
                                 Training stop.')
                break

            # save checkpoint
            if epoch % self.save_per_epochs == 0:
                self._save_checkpoints(epoch, save_best=self.improved)



    def _train_epoch(self, epoch):
        raise NotImplementedError
    def _val_epoch(self, epoch):
        raise NotImplementedError
    def _eval_metrics(self, image, mask):
        raise NotImplementedError

    def _save_checkpoints(self, epoch, save_best=False):
        
        filename = Path(self.checkpoints_path)/f'-{self.model.__class__.__name__}-{epoch}.pth'
        self.logger.info(f'\nSaving checkpoints:{filename}')
        torch.save(self.model.state_dict(), filename)

        if save_best:
            filename = self.checkpoints_path/f'{self.model.__class__.__name__}best_model.pth'
            torch.save(self.model.state_dict(), filename)
            self.logger.info(f'Saving current best: {filename}')

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint from {resume_path}')
        self.model.load_state_dict(torch.load(resume_path))


