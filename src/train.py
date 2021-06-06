import math

import cv2
import timm
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import cfg
from dataset import CassavaLeafDataModule
from loss import (CrossEntropyWithLogitsLoss, 
    EarlySmoothedFocalLoss, 
    bi_tempered_logistic_loss)


class CosineAnnealingWarmupRestarts(_LRScheduler):
   
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 T_0 : int,
                 T_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < T_0
        
        self.T0 = T_0 # first cycle step size
        self.T_mult = T_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.T_cur = T_0 # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.T_cur - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.T_cur:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.T_cur
                self.T_cur = int((self.T_cur - self.warmup_steps) * self.T_mult) + self.warmup_steps
        else:
            if epoch >= self.T0:
                if self.T_mult == 1.:
                    self.step_in_cycle = epoch % self.T0
                    self.cycle = epoch // self.T0
                else:
                    n = int(math.log((epoch / self.T0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.T0 * (self.T_mult ** n - 1) / (self.T_mult - 1))
                    self.T_cur = self.T0 * self.T_mult ** (n)
            else:
                self.T_cur = self.T0
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CassavaLeafModel(pl.LightningModule):

    def __init__(self, model_arch='tf_efficientnet_b4_ns', cfg=cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(model_name=model_arch, pretrained=True)
      
        self.model.fc = nn.Sequential(
            nn.Dropout(p=cfg.model.p_drop),
            nn.Linear(self.model.fc.in_features, cfg.model.num_classes)
        )
       
        self.model.apply(self.set_bn_eval)

        self.acc = pl.metrics.Accuracy()
        cel = CrossEntropyWithLogitsLoss() # weighting in focalloss
        self.loss_fn = bi_tempered_logistic_loss
        # self.loss_fn = EarlySmoothedFocalLoss(cel, alpha=[0.4, 0.19, 0.19, 0.03, 0.19])

    def forward(self, x):
        return self.model(x)

    def set_bn_eval(self, module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y, cfg.model.tl_t1, cfg.model.tl_t2)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y, cfg.model.tl_t1, cfg.model.tl_t2)
        acc = self.acc(y_pred, y.argmax(dim=-1))
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.acc.compute())
        for i, p in enumerate(self.optim.param_groups):
            self.log(f'lr/lr{i}', p['lr'])

    def configure_optimizers(self):
        if cfg.optim.optimizer == 'Adam':
            self.optim = torch.optim.Adam(
                self.model.parameters(), 
                lr=cfg.optim.lr,
                betas=(cfg.optim.beta1, cfg.optim.beta2),
                weight_decay=cfg.optim.weight_decay
            )
        elif cfg.optim.optimizer == 'AdamW':
            self.optim = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.optim.lr,
                betas=(cfg.optim.beta1, cfg.optim.beta2),
                weight_decay=cfg.optim.weight_decay
            )
        elif cfg.optim.optimizer == 'sgd':
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.optim.lr,
                momentum=cfg.optim.momentum,
                weight_decay=cfg.optim.weight_decay
            )
               
        if cfg.optim.scheduler == 'cosine':
            self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optim,
                T_0=cfg.optim.T0,
                T_mult=cfg.optim.T_mult,
                eta_min=cfg.optim.eta_min
            )
        elif cfg.optim.scheduler == 'cosine_warmup':
            self.sched = CosineAnnealingWarmupRestarts(
                optimizer=self.optim,
                T_0=cfg.optim.T0,
                T_mult=cfg.optim.T_mult,
                max_lr=cfg.optim.lr,
                min_lr=cfg.optim.eta_min,
                gamma=cfg.optim.gamma,
                warmup_steps=cfg.optim.warmup_steps
            )
        else:
            return self.optim

        return [self.optim], [self.sched]

def main():
    df_train = pd.read_csv('../input/train.csv')
    folds = StratifiedKFold(
        n_splits=cfg.data.num_folds, 
        shuffle=True, 
        random_state=cfg.seed).split(
            np.arange(df_train.shape[0]), df_train.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
    
        tfrec_train_path = f'../tfrecord/fold{fold}/' + 'train{}.tfrec'
        tfrec_val_path = f'../tfrecord/fold{fold}/' + 'val{}.tfrec'
        dm = CassavaLeafDataModule(df_train, trn_idx, val_idx)
        dm.setup()

        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath='../models-{}/'.format(fold),
            filename='effd4-{epoch:02d}-{val_acc:.3f}',
            save_top_k=3,
            mode='max'
        )

        logger = WandbLogger(
            project='cldc',
            name=f'fold{fold}-sgd-fmix',
            save_dir='E:\\cldc\\wandb_run'
        )

        trainer = pl.Trainer(**cfg.train,
            logger=logger, 
            callbacks=[checkpoint_callback]
        )
        
        model = CassavaLeafModel(model_arch=cfg.model.arch)
        trainer.fit(model, dm)

        break
        
        del dm, checkpoint_callback, logger, trainer, model
        torch.cuda.caching_allocator_delete()


if __name__ == "__main__":
    main()