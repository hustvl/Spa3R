import os

from .base import BaseFrameDataset


class ScanNetPP(BaseFrameDataset):

    SPLIT_FILE = {'train': 'nvs_sem_train.txt', 'val': 'nvs_sem_val.txt'}
    CAMERA_DIRS = ('dslr/undistorted_images', )  # 'iphone/rgb'

    @property
    def _cache_path(self):
        return self.data_root / 'indices' / f'scannetpp_{self.split}.pkl'

    def _load_scenes(self):
        scenes = []
        split_file = self.data_root / 'splits' / self.SPLIT_FILE[self.split]
        with open(split_file) as f:
            scene_ids = [line.strip() for line in f if line.strip()]

        for scene_id in scene_ids:
            for camera_dir in self.CAMERA_DIRS:
                image_files = os.listdir(self.data_root / 'data' / scene_id /
                                         camera_dir)
                image_files.sort()
                scene_info = {
                    'scene_id': scene_id,
                    'camera_dir': camera_dir,
                    'image_files': image_files,
                    'num_images': len(image_files)
                }
                scenes.append(scene_info)
        return scenes

    def _get_image_path(self, scene_info, image_file):
        return (self.data_root / 'data' / scene_info['scene_id'] /
                scene_info['camera_dir'] / image_file)
