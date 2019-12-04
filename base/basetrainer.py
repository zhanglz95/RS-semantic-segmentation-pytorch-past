import logging
import torch
import utils
import json
from torch.utils import tensorboard
from pathlib import Path
from datetime import datetime

# "trainer":{
#     "epochs": 80,

#     "loss": {
#         "name": "CrossEntropy",
#         "args": {}
#     },

#     "optim" : {
#         "name": "Adam", 
#         "args": {}
#     },

#     "lr_scheduler":{
#         "name":"StepLR",
#         "args":{}
#     },

#     "moniter": "mIou"

#     "tensorboard": true, 
#     "writer_dir": "", 
#     "log_per_iter": 20, 

#     "val": true, 
#     "val_per_epochs": 5, 

#     "save_per_epochs": 5,
#     "checkpoints_path":"saved/checkpoints/",
#     "brake_for_grad_vanish": 10,
    
#     "resume": {
#         "path":""
#     }
# }
def creat_object(module, name, **args):
    return getattr(module, name)(**args)

class BaseTrainer:
    def __init__(self, config, model, train_loader, val_loader=None):
        self.model = model
        self.config = config['trainer']
        
        self.epochs = config['epochs']
        self.train_loader = train_loader
        self.log_per_iter = config['log_per_iter']
        self.val = self.config['val']
        self.val_loader = val_loader
        self.val_per_epochs = config['val_per_epochs']
        
        self.logger = logging.getLogger(self.__class__.__name__)

        self.loss = creat_object(utils.loss, self.config['loss'])

        params = filter(lambda p:p.requires_grad, self.model.parameters())
        self.lr_scheduler = creat_object(utils.lr_scheduler, config['lr_sceduler'])
        self.optimizer = creat_object(utils.optim, self.config['optim'], 
                                        **{'params': params,
                                            'lr':self.lr_scheduler(0)})
        self.classes = train_loader.classes
        
        self.start_epoch = 1
        self.improved = False
        self.best = 0
        self.brake_for_grad_vanish = config['brake_for_grad_vanish']
        self.moniter = config['moniter']

        # Set Device
        self.available_gpus = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:0' if self.available_gpus else 'cpu')
        self.model.to(self.device)

        # CheckPoint and Tensorboard
        self.save_per_epochs = config['save_per_epochs']
        start_time = datetime.now().strftime("%m-%d_%H-%M")
        self.checkpoints_path = Path(self.config['checkpoints_path'])/(config['name']+str(start_time))
        self.checkpoints_path.mkdir()
        config_save_path = self.checkpoints_path/"config.json"
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=True)

        writer_dir = Path(self.config['writer_dir'])/config['name']/str(start_time)
        self.writer = tensorboard.SummaryWriter(writer_dir)

        if self.config['resume']:
            self._resume_checkpoint(self.config['resume']['resume_path'])

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
            self.improved = results[self.moniter] > self.best
            if self.improved:
                self.best = results[self.moniter]
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


