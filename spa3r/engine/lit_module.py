from collections import defaultdict

import lightning as L
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from .. import build_from_configs, models


class LitModule(L.LightningModule):

    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 criterion=None,
                 evaluator=None,
                 **kwargs):
        super().__init__()
        self.model = build_from_configs(models, model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = (
            build_from_configs(nn, criterion)
            if criterion
            else getattr(self.model, 'loss', None)
        )

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def _step(self, batch, evaluator=None):
        if isinstance(batch, tuple):
            assert len(batch) == 2
            x, y = batch
            pred = self(x)
            loss = self.criterion(pred, y)
        else:
            loss = self(batch, mode='loss')
        if evaluator:
            evaluator.update(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        if isinstance(loss, dict):
            self.log_dict(dict(loss=sum(loss.values())) | loss)
        else:
            self.log('loss', loss)
        return sum(loss.values()) if isinstance(loss, dict) else loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, self.evaluator)
        if self.evaluator:
            self.log(f'val/{self.evaluator.__class__.__name__}',
                     self.evaluator,
                     sync_dist=True)
        else:
            if isinstance(loss, dict):
                loss['val/loss'] = sum(loss.values())
                self.log_dict(loss)
            else:
                self.log('val/loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, self.evaluator)
        if self.evaluator:
            self.log(f'test/{self.evaluator.__class__.__name__}',
                     self.evaluator,
                     sync_dist=True)
        else:
            if isinstance(loss, dict):
                loss['test/loss'] = sum(loss.values())
                self.log_dict(loss)
            else:
                self.log('test/loss', loss)

    def on_train_epoch_start(self):
        if not self.trainer._accelerator_connector.use_distributed_sampler:
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        optimizer_cfg = OmegaConf.to_container(self.optimizer)
        paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        pg0 = []
        pgs = defaultdict(list)

        if paramwise_cfg:
            for pg_cfg in paramwise_cfg:
                assert pg_cfg.name not in pgs, (
                    f'Duplicate param group name: {pg_cfg.name}')
                pgs[pg_cfg.name].append(pg_cfg)

            for name, param in self.named_parameters():
                matched = False
                for pg_cfg in paramwise_cfg:
                    if 'name' in pg_cfg and pg_cfg.name in name:
                        pgs[pg_cfg.name].append(param)
                        matched = True
                        break
                if not matched:
                    pg0.append(param)
        else:
            pg0 = self.model.parameters()

        optimizer = build_from_configs(optim, optimizer_cfg, params=pg0)
        if paramwise_cfg:
            for pg in pgs.values():
                pg_cfg, *params = pg
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg.lr * pg_cfg.lr_mult
                optimizer.add_param_group({'params': params, **cfg})

        scheduler_cfg = OmegaConf.to_container(self.scheduler)
        interval = scheduler_cfg.pop('interval', 'epoch')
        scheduler = build_from_configs(
            optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': interval
            }
        }
