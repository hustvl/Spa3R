import itertools
import random

import torch
from torch.utils.data import RandomSampler, Sampler
from torch.utils.data._utils.collate import default_collate


class DynamicBatchSampler(Sampler):

    def __init__(self,
                 dataset=None,
                 sampler=None,
                 *,
                 num_imgs_per_sample,
                 num_imgs_per_gpu):
        if sampler is None:
            sampler = RandomSampler(dataset)
        self.sampler = sampler
        self.num_imgs_per_sample = num_imgs_per_sample
        self.num_imgs_per_gpu = num_imgs_per_gpu
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_iter = iter(self.sampler)
        rng = random.Random(self.epoch)

        while True:
            rand_num_imgs = rng.randint(*self.num_imgs_per_sample)
            batch_size = max(1, int(self.num_imgs_per_gpu // rand_num_imgs))

            batch = list(itertools.islice(sampler_iter, batch_size))
            if not batch:
                break
            yield [(idx, rand_num_imgs) for idx in batch]

    def __len__(self):
        avg_num_imgs = sum(self.num_imgs_per_sample) / 2
        avg_batch_size = max(1, int(self.num_imgs_per_gpu // avg_num_imgs))
        return int(len(self.sampler) // avg_batch_size)


class AlignTensorSizesCollate:

    def __call__(self, batch):
        return align_tensor_sizes_collate_fn(batch)


def align_tensor_sizes_collate_fn(batch):
    tensor_keys = []
    for key, value in batch[0].items():
        if isinstance(value, torch.Tensor):
            tensor_keys.append(key)

    for key in tensor_keys:
        for dim in range(batch[0][key].ndim):
            sizes = [item[key].size(dim) for item in batch]
            if len(set(sizes)) > 1:
                min_size = min(sizes)
                for i, item in enumerate(batch):
                    tensor = item[key]
                    ndim = tensor.ndim
                    indices = [slice(None)] * ndim
                    crop_size = sizes[i] - min_size
                    indices[dim] = slice(crop_size // 2, min_size + crop_size // 2)
                    item[key] = tensor[tuple(indices)]

    return default_collate(batch)
