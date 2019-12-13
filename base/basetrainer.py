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
        self.epochs = self.config['epochs']
        self.train_loader = train_loader
        self.val = self.config['val']
        self.val_loader = val_loader
        self.val_per_epochs = self.config['val_per_epochs']
        
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
        self.best_iou = 0

        self.loss_not_improved_count = 0
        self.iou_not_improved_count = 0
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
            this_loss = self._train_epoch(epoch)
            if self.val and epoch % self.val_per_epochs == 0:
                iou = self._val_epoch(epoch)
                self.improved = iou > self.best_iou
                if self.improved:
                    self.best_iou = iou
                    self.iou_not_improved_count = 0
                    self._save_checkpoints("best_iou")
                else:
                    self.iou_not_improved_count += self.val_per_epochs

                if self.iou_not_improved_count > self.break_for_grad_vanish * 2:
                    break
            # results = self._val_epoch(epoch)

            # Check if this is the best model
            self.improved = this_loss < self.best_loss
            if self.improved:
                self.best_loss = this_loss
                self.loss_not_improved_count = 0
                self._save_checkpoints("best_loss")
            else:
                self.loss_not_improved_count += 1

            if self.loss_not_improved_count > self.break_for_grad_vanish:
                break

            # save checkpoint
            if epoch % self.save_per_epochs == 0:
                self._save_checkpoints(epoch)



    def _train_epoch(self, epoch):
        raise NotImplementedError
    def _val_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoints(self, name):
        filename = Path(self.checkpoints_path)/f'-{self.model.__class__.__name__}-{name}.pth'
        torch.save(self.model.state_dict(), filename)

    def _resume_checkpoint(self, resume_path):
        self.model.load_state_dict(torch.load(resume_path))


