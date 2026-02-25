import logging
import os
from typing import Dict, Optional

from lightning.pytorch.loggers.csv_logs import CSVLogger, ExperimentWriter
from lightning.pytorch.loggers.logger import rank_zero_experiment

log = logging.getLogger(__name__)


class ConsoleExperimentWriter(ExperimentWriter):

    def log_metrics(
        self, metrics_dict: Dict[str, float], step: Optional[int] = None
    ) -> None:
        super().log_metrics(metrics_dict, step)
        metrics = self.metrics[-1]

        def metrics2str(metrics: dict) -> str:
            return ', '.join([
                f'{x}: {{{metrics2str(y)}}}'
                if isinstance(y, dict)
                else (
                    f'{x}: {y:.4f}' if y >= 1e-4 else f'{x}: {y:.5f}'
                    if isinstance(y, float)
                    else f'{x}: {y}'
                )
                for x, y in metrics.items()
            ])

        log_str = metrics2str(metrics)
        log.info(log_str)  # will be saved to file by Hydra

    def save(self) -> None:
        return


class ConsoleLogger(CSVLogger):

    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~lightning.pytorch.core.module.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ConsoleExperimentWriter(log_dir=self.log_dir)
        return self._experiment
