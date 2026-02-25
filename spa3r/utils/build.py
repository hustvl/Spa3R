import hydra
from hydra.core.hydra_config import HydraConfig
from lightning import pytorch as pl
from omegaconf import DictConfig, ListConfig, OmegaConf

from .console_logger import ConsoleLogger


def build_callbacks(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    loggers = [pl.loggers.TensorBoardLogger(save_dir=output_dir, name=None, version='')]
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename=f'e{{epoch}}_loss{{loss:.4f}}',
            monitor=f'loss',
            mode='min',
            save_last=True,
            auto_insert_metric_name=False)]

    if cfg.trainer.get('enable_progress_bar', True):
        callbacks.extend([
            pl.callbacks.RichModelSummary(max_depth=2),
            pl.callbacks.RichProgressBar()])
    else:
        callbacks.append(pl.callbacks.ModelSummary(max_depth=2))
        loggers.append(ConsoleLogger(save_dir=output_dir, name=None, version=''))
    return callbacks, loggers


def _instantiate_targets(cfg):
    if isinstance(cfg, (dict, DictConfig)):
        if '_target_' in cfg:
            return hydra.utils.instantiate(cfg, _convert_='all')
        return {key: _instantiate_targets(cfg[key]) for key in cfg.keys()}
    elif isinstance(cfg, (list, ListConfig)):
        return [_instantiate_targets(item) for item in cfg]
    else:
        return cfg


def build_from_configs(*args, **kwargs):
    assert len(args) in (1, 2)
    obj, cfg = args if len(args) == 2 else (None, args[0])

    if kwargs:
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.merge(cfg, kwargs)
        else:
            cfg |= kwargs

    cfg = _instantiate_targets(cfg)
    if not obj:
        return cfg
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg)
