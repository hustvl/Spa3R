import logging
import os

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from spa3r import LitDataModule, LitModule, build_callbacks

logger = logging.getLogger(__name__)


@hydra.main(config_path='../spa3r/configs', config_name='spa3r', version_base=None)
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        logger.info('Config:\n%s', OmegaConf.to_yaml(cfg))
    if 'matmul_precision' in cfg:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    data_module = LitDataModule(cfg.data)
    if 'load_from' in cfg:
        model = LitModule.load_from_checkpoint(cfg.load_from, **cfg)
    else:
        model = LitModule(**cfg)
    if cfg.get('compile', False):
        model = torch.compile(model, dynamic=True)

    callbacks, loggers = build_callbacks(cfg)
    trainer = L.Trainer(**cfg.trainer, logger=loggers, callbacks=callbacks)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
