import torch
import utils
import json
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter

def create_object(module, name, **args):
    return getattr(module, name)(**args)

class BaseTrainer:
    def __init__(self, configs, model, train_loader, val_loader=None):
        self.model = model
        self.config = configs["trainer"]
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
        self.lr = optim_config["lr"]
        self.optimizer = create_object(utils.optim, optim_config["type"], 
                                        **{'params': params,
                                            'lr':self.lr})
       # init metrics
        self.improved = False
        self.best_loss = float("inf")
        self.best_iou = 0

        self.loss_not_improved_count = 0
        self.break_for_grad_vanish = self.config['break_for_grad_vanish']
        self.lr_descend = self.config['lr_descend']

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
            json.dump(configs, handle, indent=4, sort_keys=True)
        with open(self.checkpoints_path / "message.txt", "w") as handle:
            handle.write(self.config["Message"] + "\n")

        if self.config['resume_path']:
            self._resume_checkpoint(self.config['resume_path'])

        if self.config['tensorboard']:
            self.tb_writer = SummaryWriter(log_dir=self.checkpoints_path)
        else:
            self.tb_writer = None

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            this_loss = self._train_epoch(epoch)
            validated = False
            if self.val and epoch % self.val_per_epochs == 0:
                validated = True
                iou = self._val_epoch(epoch)
                print(f"Global IoU:{iou}")
                self.improved = iou > self.best_iou
                if self.improved:
                    self.best_iou = iou
                    self._save_checkpoints("best_iou")

            # Check if this is the best model
            self.improved = this_loss < self.best_loss
            if self.improved:
                self.best_loss = this_loss
                self.loss_not_improved_count = 0
                self._save_checkpoints("best_loss")

                if self.val and not validated:
                    iou = self._val_epoch(epoch)
                    print(f"Global IoU:{iou}")
                    self.improved = iou > self.best_iou
                    if self.improved:
                        self.best_iou = iou
                        self._save_checkpoints("best_iou")
            else:
                self.loss_not_improved_count += 1

            if self.loss_not_improved_count > self.break_for_grad_vanish:
                if self.val and not validated:
                    iou = self._val_epoch(epoch)
                    print(f"Global IoU:{iou}")
                    self.improved = iou > self.best_iou
                    if self.improved:
                        self.best_iou = iou
                        self._save_checkpoints("best_iou")

                with open(self.checkpoints_path / "message.txt", "w") as handle:
                    handle.write("break for train loss best.\n")
                    handle.write(f"break in epoch {epoch}.\n")
                    handle.write(f"best_iou: {self.best_iou}\n")
                    handle.write(f"best_loss: {self.best_loss}\n")
                break
            if self.loss_not_improved_count > self.lr_descend:
                self.lr *= 0.1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

            # save checkpoint
            if epoch % self.save_per_epochs == 0:
                self._save_checkpoints(epoch)
        self.tb_writer.close()



    def _train_epoch(self, epoch):
        raise NotImplementedError
    def _val_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoints(self, name):
        filename = Path(self.checkpoints_path)/f'{self.model.__class__.__name__}-{name}.pth'
        torch.save(self.model.state_dict(), filename)

    def _resume_checkpoint(self, resume_path):
        self.model.load_state_dict(torch.load(resume_path))


