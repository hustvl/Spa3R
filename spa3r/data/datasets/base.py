import logging
import os
import pickle
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import v2 as T

from ..transforms import AdaptiveDivisibleCrop, ResizeBySide

logger = logging.getLogger(__name__)


class BaseFrameDataset(data.Dataset, ABC):

    def __init__(self,
                 data_root,
                 split=None,
                 num_frames=4,
                 sampling_interval=1,
                 target_width=518):
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.sampling_interval = sampling_interval

        self.transform = T.Compose([
            T.ToImage(),
            ResizeBySide(target_width=target_width),
            AdaptiveDivisibleCrop(divisor=14),
            T.ToDtype(torch.float32, scale=True),
        ])

        cached_file = self._cache_path
        if os.path.exists(cached_file):
            self.scenes = pickle.load(open(cached_file, 'rb'))
        else:
            self.scenes = self._load_scenes()
            pickle.dump(self.scenes, open(cached_file, 'wb'))
            logger.info(f'Scenes cached into {cached_file}')
        logger.info(f'Loaded {len(self.scenes)} scenes for {self.__class__.__name__}')

    @property
    @abstractmethod
    def _cache_path(self) -> Path:
        pass

    @abstractmethod
    def _load_scenes(self) -> List[Dict]:
        pass

    @abstractmethod
    def _get_image_path(self, scene_info: Dict, image_file: str) -> Path:
        pass

    def _sample_frames(self, scene_info, num_frames=None):
        num_images = scene_info['num_images']
        if num_frames is None:
            num_frames = (random.randint(*self.num_frames)
                          if isinstance(self.num_frames, list)
                          else self.num_frames)
        sampling_interval = (self.sampling_interval
                             if isinstance(self.sampling_interval, list)
                             else [self.sampling_interval, self.sampling_interval])

        indices = []
        if num_images > (num_frames - 1) * sampling_interval[0]:
            sampling_interval[1] = min((num_images - 1) // (num_frames - 1),
                                       sampling_interval[1])
            current_idx = random.randint(
                0, num_images - (num_frames - 1) * sampling_interval[1] - 1)
            indices.append(current_idx)

            for _ in range(num_frames - 1):
                current_idx += random.randint(*sampling_interval)
                indices.append(current_idx)
        elif num_images < num_frames:
            # logger.warning(f"{scene_info['scene_id']}: {num_images} images < "
            #                f'{num_frames} frames for sampling')
            indices = random.choices(range(num_images), k=num_frames)
            indices.sort()
        else:
            # logger.info(f"{scene_info['scene_id']}: {num_frames} frames with "
            #             f'{sampling_interval[1]} interval exceeds {num_images} images')
            indices = random.sample(range(num_images), num_frames)
            indices.sort()

        return indices

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        num_frames = None
        if isinstance(idx, tuple):
            idx, num_frames = idx

        scene_info = self.scenes[idx]
        frame_indices = self._sample_frames(scene_info, num_frames=num_frames)
        images = []
        image_paths = []
        for frame_idx in frame_indices:
            image_file = scene_info['image_files'][frame_idx]
            image_path = self._get_image_path(scene_info, image_file)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            images.append(image)
            image_paths.append(image_file)

        return {
            'scene_id': scene_info['scene_id'],
            'images': torch.stack(images),
            'image_paths': image_paths,
        }
