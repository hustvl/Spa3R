import logging
from functools import partial

import lightning as L
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data.distributed import DistributedSampler

from .. import build_from_configs
from . import datasets

logger = logging.getLogger(__name__)


def build_data_loaders(cfg: DictConfig):
    """Deprecated: Use LitDataModule instead."""
    if 'splits' in cfg.dataset:
        if isinstance(cfg, DictConfig):
            with open_dict(cfg):
                split_cfgs = cfg.dataset.pop('splits')
        if isinstance(split_cfgs, ListConfig):
            split_cfgs = {split: {'split': split} for split in split_cfgs}
    else:
        split_cfgs = {'default': {}}

    def _build_data_loader(split_cfg):
        with open_dict(cfg):
            split_cfg = OmegaConf.merge(cfg, dict(dataset=split_cfg))
        split_cfg._set_flag('allow_objects', True)

        if 'batch_sampler' in split_cfg:
            split_cfg.dataset = build_from_configs(datasets, split_cfg.dataset)
            split_cfg.batch_sampler = build_from_configs(
                split_cfg.batch_sampler, dataset=split_cfg.dataset)
        if 'type' not in split_cfg and '_target_' not in split_cfg:
            split_cfg._target_ = 'torch.utils.data.DataLoader'
        return build_from_configs(split_cfg)

    return [_build_data_loader(cfg_) for split, cfg_ in split_cfgs.items()]


class LitDataModule(L.LightningDataModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.split_cfgs = self._parse_split_configs(cfg.dataset)

        self.train_dataloader = partial(self._build_data_loader, 'train')
        if 'val' in self.split_cfgs:
            self.val_dataloader = partial(self._build_data_loader, 'val')
        if 'test' in self.split_cfgs:
            self.test_dataloader = partial(self._build_data_loader, 'test')

    def _parse_split_configs(self, dataset_cfg):
        if 'splits' in dataset_cfg:
            with open_dict(dataset_cfg):
                split_cfgs = dataset_cfg.pop('splits')
            if isinstance(split_cfgs, (ListConfig)):
                split_cfgs = {split: {'split': split} for split in split_cfgs}
            return split_cfgs
        return {'default': {}}

    def _build_data_loader(self, split):
        split_cfg = self.split_cfgs.get(split, {})
        with open_dict(self.cfg):
            split_cfg = OmegaConf.merge(self.cfg, dict(dataset=split_cfg))
        split_cfg._set_flag('allow_objects', True)

        split_cfg.dataset = build_from_configs(datasets, split_cfg.dataset)
        if 'batch_sampler' in split_cfg:
            if dist.is_initialized():
                split_cfg.batch_sampler.sampler = DistributedSampler(split_cfg.dataset)
                if dist.get_rank() == 0:
                    logger.info('Switching to DistributedSampler for DDP')

            split_cfg.batch_sampler = build_from_configs(
                split_cfg.batch_sampler, dataset=split_cfg.dataset)

        if 'type' not in split_cfg and '_target_' not in split_cfg:
            split_cfg._target_ = 'torch.utils.data.DataLoader'
        return build_from_configs(split_cfg)
