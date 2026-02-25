import bisect

from torch.utils import data


class ConcatDataset(data.ConcatDataset):

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, *args = idx
        else:
            args = None

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        if args is None:
            return self.datasets[dataset_idx][sample_idx]
        else:
            return self.datasets[dataset_idx][(sample_idx, *args)]
